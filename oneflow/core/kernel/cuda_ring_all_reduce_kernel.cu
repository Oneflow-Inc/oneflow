#include "oneflow/core/kernel/cuda_ring_all_reduce_kernel.h"
#include "oneflow/core/kernel/kernel_util.cuh"
#include <device_launch_parameters.h>
#include "oneflow/core/common/reduce_method.pb.h"

namespace oneflow {

namespace {

using Pack = ulong2;

constexpr int32_t PACK_SIZE = sizeof(Pack);
constexpr int32_t PACK_ALIGN = alignof(Pack);
constexpr int32_t NUM_WARP_PER_BLOCK = 8;
constexpr int32_t NUM_THREAD_PER_WARP = 32;
constexpr int32_t NUM_THREAD = NUM_THREAD_PER_WARP * NUM_WARP_PER_BLOCK;
constexpr int32_t NUM_PACK_PER_BATCH_PER_THREAD = 8;
constexpr int32_t NUM_BLOCK_PER_LINK = 2;

__forceinline__ __device__ int64_t DivUp(int64_t n, int64_t val) { return (n + val - 1) / val; }

template<ReduceMethod method, typename T>
struct ReduceFunctor {
  __device__ __forceinline__ T operator()(const T& a, const T& b) const;
};

template<typename T>
struct ReduceFunctor<ReduceMethod::kSum, T> {
  __device__ __forceinline__ T operator()(const T& a, const T& b) const { return a + b; }
};

template<typename T>
struct ReduceFunctor<ReduceMethod::kProd, T> {
  __device__ __forceinline__ T operator()(const T& a, const T& b) const { return a * b; }
};

template<typename T>
struct ReduceFunctor<ReduceMethod::kMax, T> {
  __device__ __forceinline__ T operator()(const T& a, const T& b) const { return max(a, b); }
};

template<typename T>
struct ReduceFunctor<ReduceMethod::kMin, T> {
  __device__ __forceinline__ T operator()(const T& a, const T& b) const { return min(a, b); }
};

template<ReduceMethod method, typename T, typename P>
struct PackReduceFunctor {
  static_assert(sizeof(P) % sizeof(T) == 0,
                "The size of the P must be a multiple of the size of T");
  union View {
    P p;
    T t[sizeof(P) / sizeof(T)];
  };
  __device__ __forceinline__ P operator()(const P& a, const P& b) const {
    View va;
    View vb;
    View vc;
    va.p = a;
    vb.p = b;
#pragma unroll
    for (size_t i = 0; i < sizeof(P) / sizeof(T); ++i) {
      vc.t[i] = ReduceFunctor<method, T>()(va.t[i], vb.t[i]);
    }
    return vc.p;
  }
};

template<ReduceMethod method>
struct PackReduceFunctor<method, Pack, float> {
  union View64 {
    ulong p;
    float2 f2;
  };
  __device__ __forceinline__ Pack operator()(const Pack& a, const Pack& b) const {
    Pack res;
    View64 va;
    View64 vb;
    va.p = a.x;
    vb.p = b.x;
    va.f2.x += vb.f2.x;
    va.f2.y += vb.f2.y;
    res.x = va.p;
    va.p = a.y;
    vb.p = b.y;
    va.f2.x += vb.f2.x;
    va.f2.y += vb.f2.y;
    res.y = va.p;
    return res;
  }
};

template<ReduceMethod method, typename T>
struct PackReduceFunctor<method, T, T> {
  __device__ __forceinline__ T operator()(const T& a, const T& b) const {
    return ReduceFunctor<method, T>()(a, b);
  }
};

template<ReduceMethod method, typename T, typename P, int32_t BATCH>
struct BatchPackReduceFunctor {
  __device__ __forceinline__ void operator()(P (&res)[BATCH], const P (&a)[BATCH],
                                             const P (&b)[BATCH]) {
#pragma unroll
    for (int32_t i = 0; i < BATCH; ++i) { res[i] = PackReduceFunctor<method, T, P>()(a[i], b[i]); }
  }
};

template<typename T>
struct FetchFunctor {
  __device__ __forceinline__ void operator()(T& v, const T* p) { v = *p; }
};

template<typename T>
struct StoreFunctor {
  __device__ __forceinline__ void operator()(T* p, const T& v) { *p = v; }
};

template<>
struct FetchFunctor<Pack> {
  __device__ __forceinline__ void operator()(Pack& v, const Pack* p) {
    asm volatile("ld.volatile.global.v2.u64 {%0,%1}, [%2];"
                 : "=l"(v.x), "=l"(v.y)
                 : "l"(p)
                 : "memory");
  }
};

template<>
struct StoreFunctor<Pack> {
  __device__ __forceinline__ void operator()(Pack* p, const Pack& v) {
    asm volatile("st.volatile.global.v2.u64 [%0], {%1,%2};" ::"l"(p), "l"(v.x), "l"(v.y)
                 : "memory");
  }
};

template<typename T, int32_t BATCH, int32_t STRIDE, bool BOUND>
struct BatchFetchFunctor {
  __device__ __forceinline__ void operator()(T (&v)[BATCH], const T* start, const T* bound) {
#pragma unroll
    for (int32_t i = 0; i < BATCH; ++i) {
      const T* ptr = start + i * STRIDE;
      if (!BOUND || ptr < bound) { FetchFunctor<T>()(v[i], ptr); }
    }
  }
};

template<typename T, int32_t BATCH, int32_t STRIDE, bool BOUND>
struct BatchStoreFunctor {
  __device__ __forceinline__ void operator()(T* start, T (&v)[BATCH], const T* bound) {
#pragma unroll
    for (int32_t i = 0; i < BATCH; ++i) {
      T* ptr = start + i * STRIDE;
      if (!BOUND || ptr < bound) { StoreFunctor<T>()(ptr, v[i]); }
    }
  }
};

template<ReduceMethod method, typename T, typename P, int32_t BATCH, bool BOUND, int32_t NUM_IN,
         int32_t NUM_OUT>
__device__ __forceinline__ void BatchPackReduceOrCopy(const int64_t num_elem,
                                                      const T* (&in)[NUM_IN], T* (&out)[NUM_OUT]) {
  constexpr int32_t NUM_PACK_PER_BATCH_PER_WARP = BATCH * NUM_THREAD_PER_WARP;
  constexpr int32_t NUM_ELEM_PER_PACK = sizeof(P) / sizeof(T);
  constexpr int32_t NUM_PACK_PER_BATCH_PER_BLOCK = NUM_PACK_PER_BATCH_PER_WARP * NUM_WARP_PER_BLOCK;
  const int32_t thread_id = threadIdx.x;
  const int32_t warp_id = thread_id / NUM_THREAD_PER_WARP;
  const int32_t lane_id = thread_id % NUM_THREAD_PER_WARP;
  const int32_t offset = warp_id * NUM_PACK_PER_BATCH_PER_WARP + lane_id;
  assert(num_elem % NUM_ELEM_PER_PACK == 0);
  const int64_t num_pack = num_elem / NUM_ELEM_PER_PACK;
  if (!BOUND) { assert(num_pack % NUM_PACK_PER_BATCH_PER_BLOCK == 0); }
  const int64_t num_batch = DivUp(num_pack, NUM_PACK_PER_BATCH_PER_BLOCK);
  const P* in_pack[NUM_IN];
  const P* in_bound[NUM_IN];
  for (int32_t i = 0; i < NUM_IN; ++i) {
    in_pack[i] = reinterpret_cast<const P*>(in[i]) + offset;
    in_bound[i] = reinterpret_cast<const P*>(in[i]) + num_pack;
  }
  P* out_pack[NUM_OUT];
  const P* out_bound[NUM_OUT];
  for (int32_t i = 0; i < NUM_OUT; ++i) {
    out_pack[i] = reinterpret_cast<P*>(out[i]) + offset;
    out_bound[i] = reinterpret_cast<const P*>(out[i]) + num_pack;
  }
  P batch[BATCH];
  using BatchPackFetch = BatchFetchFunctor<P, BATCH, NUM_THREAD_PER_WARP, BOUND>;
  using BatchPackStore = BatchStoreFunctor<P, BATCH, NUM_THREAD_PER_WARP, BOUND>;
  using BatchPackReduce = BatchPackReduceFunctor<method, T, P, BATCH>;
  for (int64_t b = 0; b < num_batch; ++b) {
    BatchPackFetch()(batch, in_pack[0], in_bound[0]);
#pragma unroll
    for (int32_t i = 1; i < NUM_IN; ++i) {
      P tmp[BATCH];
      BatchPackFetch()(tmp, in_pack[i], in_bound[i]);
      BatchPackReduce()(batch, batch, tmp);
    }
#pragma unroll
    for (int32_t i = 0; i < NUM_OUT; ++i) { BatchPackStore()(out_pack[i], batch, out_bound[i]); }
#pragma unroll
    for (int32_t i = 0; i < NUM_IN; ++i) { in_pack[i] += NUM_PACK_PER_BATCH_PER_BLOCK; }
#pragma unroll
    for (int32_t i = 0; i < NUM_OUT; ++i) { out_pack[i] += NUM_PACK_PER_BATCH_PER_BLOCK; }
  }
}

template<ReduceMethod method, typename T, typename P, int32_t BATCH, bool BOUND, int32_t NUM_IN,
         int32_t NUM_OUT>
__device__ __forceinline__ void DoBatchPackReduceOrCopy(const int64_t num_elem,
                                                        const T* (&in)[NUM_IN],
                                                        T* (&out)[NUM_OUT]) {
  BatchPackReduceOrCopy<method, T, P, BATCH, BOUND, NUM_IN, NUM_OUT>(num_elem, in, out);
  for (int32_t i = 0; i < NUM_IN; ++i) { in[i] += num_elem; }
  for (int32_t i = 0; i < NUM_OUT; ++i) { out[i] += num_elem; }
}

template<ReduceMethod method, typename T, int32_t NUM_IN, int32_t NUM_OUT>
__device__ __forceinline__ void ReduceOrCopy(const int64_t num_elem, const T* (&in)[NUM_IN],
                                             T* (&out)[NUM_OUT]) {
  bool all_same_aligned = true;
  int32_t align = reinterpret_cast<uintptr_t>(in[0]) % sizeof(Pack);
  for (int32_t i = 1; i < NUM_IN; ++i) {
    if (reinterpret_cast<uintptr_t>(in[i]) % sizeof(Pack) != align) {
      all_same_aligned = false;
      break;
    }
  }
  if (all_same_aligned) {
    for (int32_t i = 0; i < NUM_OUT; ++i) {
      if (reinterpret_cast<uintptr_t>(out[i]) % sizeof(Pack) != align) {
        all_same_aligned = false;
        break;
      }
    }
  }
  if (all_same_aligned) {
    int64_t remaining = num_elem;
    if (align > 0) {
      const int32_t num_align_elem = align / sizeof(T);
      DoBatchPackReduceOrCopy<method, T, T, NUM_PACK_PER_BATCH_PER_THREAD, true, NUM_IN, NUM_OUT>(
          num_align_elem, in, out);
      remaining -= num_align_elem;
    }
    constexpr int32_t NUM_ELEM_PER_BATCH =
        sizeof(Pack) / sizeof(T) * NUM_PACK_PER_BATCH_PER_THREAD * NUM_THREAD;
    const int64_t num_batch = remaining / NUM_ELEM_PER_BATCH;
    if (num_batch > 0) {
      const int64_t total_batch_elem = num_batch * NUM_ELEM_PER_BATCH;
      DoBatchPackReduceOrCopy<method, T, Pack, NUM_PACK_PER_BATCH_PER_THREAD, false, NUM_IN,
                              NUM_OUT>(total_batch_elem, in, out);
      remaining -= total_batch_elem;
    }
    if (remaining > 0) {
      DoBatchPackReduceOrCopy<method, T, T, NUM_PACK_PER_BATCH_PER_THREAD, true, NUM_IN, NUM_OUT>(
          remaining, in, out);
    }
  } else {
    int64_t remaining = num_elem;
    constexpr int32_t NUM_ELEM_PER_BATCH = NUM_PACK_PER_BATCH_PER_THREAD * NUM_THREAD;
    const int64_t num_batch = remaining / NUM_ELEM_PER_BATCH;
    if (num_batch > 0) {
      const int64_t total_batch_elem = num_batch * NUM_ELEM_PER_BATCH;
      DoBatchPackReduceOrCopy<method, T, T, NUM_PACK_PER_BATCH_PER_THREAD, false, NUM_IN, NUM_OUT>(
          total_batch_elem, in, out);
      remaining -= total_batch_elem;
    } else {
      DoBatchPackReduceOrCopy<method, T, T, NUM_PACK_PER_BATCH_PER_THREAD, true, NUM_IN, NUM_OUT>(
          remaining, in, out);
    }
  }
}

template<ReduceMethod method, typename T, bool RECV, bool SRC, bool SEND, bool DST>
__global__ void GenericOp(CudaRingAllReduceParams<T> params) {
  const int32_t block_id = blockIdx.x;
  const int32_t link_id = block_id / NUM_BLOCK_PER_LINK;
  const CudaRingAllReduceLinkParams<T>& link_params = params.links[link_id];
  const int32_t block_id_in_link = block_id % NUM_BLOCK_PER_LINK;
  const int64_t num_elem_per_block = DivUp(link_params.num_elem, NUM_BLOCK_PER_LINK);
  const int64_t block_offset = block_id_in_link * num_elem_per_block;
  const int64_t block_num_elem = min(num_elem_per_block, link_params.num_elem - block_offset);
  if (block_num_elem > 0) {
    constexpr int32_t NUM_IN = RECV + SRC;
    const T* in[NUM_IN];
    if (RECV) {
      in[0] = link_params.recv + block_offset;
      if (SRC) { in[1] = link_params.src + block_offset; }
    } else {
      in[0] = link_params.src + block_offset;
    }
    constexpr int32_t NUM_OUT = SEND + DST;
    T* out[NUM_OUT];
    if (SEND) {
      out[0] = link_params.send + block_offset;
      if (DST) { out[1] = link_params.dst + block_offset; }
    } else {
      out[0] = link_params.dst + block_offset;
    }
    ReduceOrCopy<method, T, NUM_IN, NUM_OUT>(block_num_elem, in, out);
  }
}

template<ReduceMethod method, typename T, bool RECV, bool SRC, bool SEND, bool DST>
void LaunchGenericOp(DeviceCtx* ctx, const CudaRingAllReduceParams<T>& params) {
  GenericOp<method, T, RECV, SRC, SEND, DST>
      <<<params.num_links * NUM_BLOCK_PER_LINK, NUM_THREAD, 0, ctx->cuda_stream()>>>(params);
}

}  // namespace

template<typename T>
void CudaRingAllReduceKernelUtil<T>::Send(DeviceCtx* ctx, CudaRingAllReduceParams<T> params) {
  LaunchGenericOp<ReduceMethod::kSum, T, false, true, true, false>(ctx, params);
}

template<typename T>
void CudaRingAllReduceKernelUtil<T>::RecvReduceSend(DeviceCtx* ctx,
                                                    CudaRingAllReduceParams<T> params) {
  LaunchGenericOp<ReduceMethod::kSum, T, true, true, true, false>(ctx, params);
}

template<typename T>
void CudaRingAllReduceKernelUtil<T>::RecvReduceSendCopy(DeviceCtx* ctx,
                                                        CudaRingAllReduceParams<T> params) {
  LaunchGenericOp<ReduceMethod::kSum, T, true, true, true, true>(ctx, params);
}

template<typename T>
void CudaRingAllReduceKernelUtil<T>::RecvSendCopy(DeviceCtx* ctx,
                                                  CudaRingAllReduceParams<T> params) {
  LaunchGenericOp<ReduceMethod::kSum, T, true, false, true, true>(ctx, params);
}

template<typename T>
void CudaRingAllReduceKernelUtil<T>::RecvCopy(DeviceCtx* ctx, CudaRingAllReduceParams<T> params) {
  LaunchGenericOp<ReduceMethod::kSum, T, true, false, false, true>(ctx, params);
}

#define INSTANTIATE_CUDA_RING_ALL_REDUCE_KERNEL_UTIL(type_cpp, type_proto) \
  template struct CudaRingAllReduceKernelUtil<type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_CUDA_RING_ALL_REDUCE_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
