#include "oneflow/core/kernel/cuda_ring_all_reduce_kernel.h"
#include "oneflow/core/kernel/kernel_util.cuh"
#include <device_launch_parameters.h>
#include "oneflow/core/common/reduce_method.pb.h"

namespace oneflow {

constexpr int32_t PACK_SIZE = sizeof(ulong2);
constexpr int32_t PACK_ALIGN = alignof(ulong2);
constexpr int32_t NUM_WARP = 8;
constexpr int32_t NUM_THREAD_PER_WARP = 32;
constexpr int32_t NUM_THREAD = NUM_THREAD_PER_WARP * NUM_WARP;
constexpr int32_t NUM_LINE_PER_CHUNK = 32;
constexpr int32_t NUM_PACK_PER_LINE_PER_THREAD = 8;
constexpr int32_t NUM_PACK_PER_LINE_PER_WARP = NUM_PACK_PER_LINE_PER_THREAD * NUM_THREAD_PER_WARP;
constexpr int32_t NUM_PACK_PER_LINE = NUM_PACK_PER_LINE_PER_WARP * NUM_WARP;
constexpr int32_t LINE_SIZE = NUM_PACK_PER_LINE * PACK_SIZE;

namespace {

using Pack = ulong2;

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

template<ReduceMethod method, typename T>
struct PackReduceFunctor {
  static_assert(sizeof(Pack) % sizeof(T) == 0,
                "The size of the Pack must be a multiple of the size of T");
  union View {
    Pack p;
    T t[sizeof(Pack) / sizeof(T)];
  };
  __device__ __forceinline__ Pack operator()(const Pack& a, const Pack& b) const {
    View va;
    View vb;
    View vc;
    va.p = a;
    vb.p = b;
#pragma unroll
    for (size_t i = 0; i < sizeof(Pack) / sizeof(T); ++i) {
      vc.t[i] = ReduceFunctor<method, T>()(va.t[i], vb.t[i]);
    }
    return vc.p;
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

template<ReduceMethod method, typename T, bool RECV, bool SRC, bool SEND, bool DST>
__device__ void AlignedReduceOrCopy(const int64_t num_elem, const T* recv, const T* src, T* send,
                                    T* dst) {
  const int32_t thread_id = threadIdx.x;
  const int32_t num_elem_per_line = LINE_SIZE / sizeof(T);
  const int64_t num_line = num_elem / num_elem_per_line;
  const int32_t warp_id = thread_id / NUM_THREAD_PER_WARP;
  const int32_t lane_id = thread_id % NUM_THREAD_PER_WARP;
  const int32_t offset = warp_id * NUM_PACK_PER_LINE_PER_WARP + lane_id;
  const Pack* recv_pack_ptr = RECV ? reinterpret_cast<const Pack*>(recv) + offset : nullptr;
  const Pack* src_pack_ptr = SRC ? reinterpret_cast<const Pack*>(src) + offset : nullptr;
  Pack* send_pack_ptr = SEND ? reinterpret_cast<Pack*>(send) + offset : nullptr;
  Pack* dst_pack_ptr = DST ? reinterpret_cast<Pack*>(dst) + offset : nullptr;
  Pack line_recv[NUM_PACK_PER_LINE_PER_THREAD];
  for (int64_t l = 0; l < num_line; ++l) {
    if (RECV) {
#pragma unroll
      for (int32_t p = 0; p < NUM_PACK_PER_LINE_PER_THREAD; ++p) {
        FetchFunctor<Pack>()(line_recv[p], recv_pack_ptr + p * NUM_THREAD_PER_WARP);
      }
    }
    if (SRC) {
      if (!RECV) {
#pragma unroll
        for (int32_t p = 0; p < NUM_PACK_PER_LINE_PER_THREAD; ++p) {
          FetchFunctor<Pack>()(line_recv[p], src_pack_ptr + p * NUM_THREAD_PER_WARP);
        }
      } else {
        Pack line_src[NUM_PACK_PER_LINE_PER_THREAD];
#pragma unroll
        for (int32_t p = 0; p < NUM_PACK_PER_LINE_PER_THREAD; ++p) {
          FetchFunctor<Pack>()(line_src[p], src_pack_ptr + p * NUM_THREAD_PER_WARP);
        }
#pragma unroll
        for (int32_t p = 0; p < NUM_PACK_PER_LINE_PER_THREAD; ++p) {
          line_recv[p] = PackReduceFunctor<method, T>()(line_recv[p], line_src[p]);
        }
      }
    }
    if (SEND) {
#pragma unroll
      for (int32_t p = 0; p < NUM_PACK_PER_LINE_PER_THREAD; ++p) {
        StoreFunctor<Pack>()(send_pack_ptr + p * NUM_THREAD_PER_WARP, line_recv[p]);
      }
    }
    if (DST) {
#pragma unroll
      for (int32_t p = 0; p < NUM_PACK_PER_LINE_PER_THREAD; ++p) {
        StoreFunctor<Pack>()(dst_pack_ptr + p * NUM_THREAD_PER_WARP, line_recv[p]);
      }
    }
    if (RECV) { recv_pack_ptr += NUM_PACK_PER_LINE; }
    if (SRC) { src_pack_ptr += NUM_PACK_PER_LINE; }
    if (SEND) { send_pack_ptr += NUM_PACK_PER_LINE; }
    if (DST) { dst_pack_ptr += NUM_PACK_PER_LINE; }
  }
}

template<typename T>
__global__ void SendGpu(CudaRingAllReduceArg<T> arg) {
  const int32_t thread_id = threadIdx.x;
  const int32_t block_id = blockIdx.x;
  T* send = arg.send[block_id];
  const T* src = arg.src[block_id];
  const int64_t num_elem = arg.num_elem[block_id];
  static_assert(PACK_SIZE % sizeof(T) == 0, "");
  const int32_t num_elem_per_line = LINE_SIZE / sizeof(T);
  const int64_t num_line = num_elem / num_elem_per_line;
  const int32_t warp_id = thread_id / NUM_THREAD_PER_WARP;
  const int32_t lane_id = thread_id % NUM_THREAD_PER_WARP;
  ulong2 line[NUM_PACK_PER_LINE_PER_THREAD];
  const int32_t offset = warp_id * NUM_PACK_PER_LINE_PER_WARP + lane_id;
  ulong2* send_pack_ptr = reinterpret_cast<ulong2*>(send) + offset;
  const ulong2* src_pack_ptr = reinterpret_cast<const ulong2*>(src) + offset;
  for (int64_t l = 0; l < num_line; ++l) {
#pragma unroll
    for (int32_t p = 0; p < NUM_PACK_PER_LINE_PER_THREAD; ++p) {
      line[p] = *(src_pack_ptr + p * NUM_THREAD_PER_WARP);
    }
#pragma unroll
    for (int32_t p = 0; p < NUM_PACK_PER_LINE_PER_THREAD; ++p) {
      *(send_pack_ptr + p * NUM_THREAD_PER_WARP) = line[p];
    }
    send_pack_ptr += NUM_PACK_PER_LINE;
    src_pack_ptr += NUM_PACK_PER_LINE;
  }
}

template<typename T>
__global__ void RecvReduceSendGpu(CudaRingAllReduceArg<T> arg) {
  const int32_t thread_id = threadIdx.x;
  const int32_t block_id = blockIdx.x;
  T* send = arg.send[block_id];
  const T* src = arg.src[block_id];
  const T* recv = arg.recv[block_id];
  const int64_t num_elem = arg.num_elem[block_id];
  static_assert(PACK_SIZE % sizeof(T) == 0, "");
  const int32_t num_elem_per_line = LINE_SIZE / sizeof(T);
  const int64_t num_line = num_elem / num_elem_per_line;
  const int32_t warp_id = thread_id / NUM_THREAD_PER_WARP;
  const int32_t lane_id = thread_id % NUM_THREAD_PER_WARP;
  ulong2 line_recv[NUM_PACK_PER_LINE_PER_THREAD];
  ulong2 line_src[NUM_PACK_PER_LINE_PER_THREAD];
  const int32_t offset = warp_id * NUM_PACK_PER_LINE_PER_WARP + lane_id;
  ulong2* send_pack_ptr = reinterpret_cast<ulong2*>(send) + offset;
  const ulong2* src_pack_ptr = reinterpret_cast<const ulong2*>(src) + offset;
  const ulong2* recv_pack_ptr = reinterpret_cast<const ulong2*>(recv) + offset;
  for (int64_t l = 0; l < num_line; ++l) {
#pragma unroll
    for (int32_t p = 0; p < NUM_PACK_PER_LINE_PER_THREAD; ++p) {
      FetchFunctor<Pack>()(line_recv[p], recv_pack_ptr + p * NUM_THREAD_PER_WARP);
    }
#pragma unroll
    for (int32_t p = 0; p < NUM_PACK_PER_LINE_PER_THREAD; ++p) {
      FetchFunctor<Pack>()(line_src[p], src_pack_ptr + p * NUM_THREAD_PER_WARP);
    }
#pragma unroll
    for (int32_t p = 0; p < NUM_PACK_PER_LINE_PER_THREAD; ++p) {
      line_recv[p] = PackReduceFunctor<ReduceMethod::kSum, T>()(line_recv[p], line_src[p]);
    }
#pragma unroll
    for (int32_t p = 0; p < NUM_PACK_PER_LINE_PER_THREAD; ++p) {
      StoreFunctor<Pack>()(send_pack_ptr + p * NUM_THREAD_PER_WARP, line_recv[p]);
    }
    send_pack_ptr += NUM_PACK_PER_LINE;
    src_pack_ptr += NUM_PACK_PER_LINE;
    recv_pack_ptr += NUM_PACK_PER_LINE;
  }
}

template<typename T>
__global__ void RecvReduceSendCopyGpu(CudaRingAllReduceArg<T> arg) {
  const int32_t thread_id = threadIdx.x;
  const int32_t block_id = blockIdx.x;
  T* send = arg.send[block_id];
  const T* recv = arg.recv[block_id];
  T* dst = arg.dst[block_id];
  const T* src = arg.src[block_id];
  const int64_t num_elem = arg.num_elem[block_id];
  static_assert(PACK_SIZE % sizeof(T) == 0, "");
  const int32_t num_elem_per_line = LINE_SIZE / sizeof(T);
  const int64_t num_line = num_elem / num_elem_per_line;
  const int32_t warp_id = thread_id / NUM_THREAD_PER_WARP;
  const int32_t lane_id = thread_id % NUM_THREAD_PER_WARP;
  ulong2 line_recv[NUM_PACK_PER_LINE_PER_THREAD];
  ulong2 line_src[NUM_PACK_PER_LINE_PER_THREAD];
  const int32_t offset = warp_id * NUM_PACK_PER_LINE_PER_WARP + lane_id;
  ulong2* send_pack_ptr = reinterpret_cast<ulong2*>(send) + offset;
  ulong2* dst_pack_ptr = reinterpret_cast<ulong2*>(dst) + offset;
  const ulong2* src_pack_ptr = reinterpret_cast<const ulong2*>(src) + offset;
  const ulong2* recv_pack_ptr = reinterpret_cast<const ulong2*>(recv) + offset;
  for (int64_t l = 0; l < num_line; ++l) {
#pragma unroll
    for (int32_t p = 0; p < NUM_PACK_PER_LINE_PER_THREAD; ++p) {
      FetchFunctor<Pack>()(line_recv[p], recv_pack_ptr + p * NUM_THREAD_PER_WARP);
    }
#pragma unroll
    for (int32_t p = 0; p < NUM_PACK_PER_LINE_PER_THREAD; ++p) {
      FetchFunctor<Pack>()(line_src[p], src_pack_ptr + p * NUM_THREAD_PER_WARP);
    }
#pragma unroll
    for (int32_t p = 0; p < NUM_PACK_PER_LINE_PER_THREAD; ++p) {
      line_recv[p] = PackReduceFunctor<ReduceMethod::kSum, T>()(line_recv[p], line_src[p]);
    }
#pragma unroll
    for (int32_t p = 0; p < NUM_PACK_PER_LINE_PER_THREAD; ++p) {
      StoreFunctor<Pack>()(send_pack_ptr + p * NUM_THREAD_PER_WARP, line_recv[p]);
    }
#pragma unroll
    for (int32_t p = 0; p < NUM_PACK_PER_LINE_PER_THREAD; ++p) {
      StoreFunctor<Pack>()(dst_pack_ptr + p * NUM_THREAD_PER_WARP, line_recv[p]);
    }
    send_pack_ptr += NUM_PACK_PER_LINE;
    dst_pack_ptr += NUM_PACK_PER_LINE;
    src_pack_ptr += NUM_PACK_PER_LINE;
    recv_pack_ptr += NUM_PACK_PER_LINE;
  }
}

template<typename T>
__global__ void RecvSendCopyGpu(CudaRingAllReduceArg<T> arg) {
  const int32_t thread_id = threadIdx.x;
  const int32_t block_id = blockIdx.x;
  T* send = arg.send[block_id];
  const T* recv = arg.recv[block_id];
  T* dst = arg.dst[block_id];
  const int64_t num_elem = arg.num_elem[block_id];
  static_assert(PACK_SIZE % sizeof(T) == 0, "");
  const int32_t num_elem_per_line = LINE_SIZE / sizeof(T);
  const int64_t num_line = num_elem / num_elem_per_line;
  const int32_t warp_id = thread_id / NUM_THREAD_PER_WARP;
  const int32_t lane_id = thread_id % NUM_THREAD_PER_WARP;
  ulong2 line_recv[NUM_PACK_PER_LINE_PER_THREAD];
  const int32_t offset = warp_id * NUM_PACK_PER_LINE_PER_WARP + lane_id;
  ulong2* send_pack_ptr = reinterpret_cast<ulong2*>(send) + offset;
  ulong2* dst_pack_ptr = reinterpret_cast<ulong2*>(dst) + offset;
  const ulong2* recv_pack_ptr = reinterpret_cast<const ulong2*>(recv) + offset;
  for (int64_t l = 0; l < num_line; ++l) {
#pragma unroll
    for (int32_t p = 0; p < NUM_PACK_PER_LINE_PER_THREAD; ++p) {
      FetchFunctor<Pack>()(line_recv[p], recv_pack_ptr + p * NUM_THREAD_PER_WARP);
    }
#pragma unroll
    for (int32_t p = 0; p < NUM_PACK_PER_LINE_PER_THREAD; ++p) {
      StoreFunctor<Pack>()(send_pack_ptr + p * NUM_THREAD_PER_WARP, line_recv[p]);
    }
#pragma unroll
    for (int32_t p = 0; p < NUM_PACK_PER_LINE_PER_THREAD; ++p) {
      StoreFunctor<Pack>()(dst_pack_ptr + p * NUM_THREAD_PER_WARP, line_recv[p]);
    }
    send_pack_ptr += NUM_PACK_PER_LINE;
    dst_pack_ptr += NUM_PACK_PER_LINE;
    recv_pack_ptr += NUM_PACK_PER_LINE;
  }
}

template<typename T>
__global__ void RecvCopyGpu(CudaRingAllReduceArg<T> arg) {
  const int32_t thread_id = threadIdx.x;
  const int32_t block_id = blockIdx.x;
  const T* recv = arg.recv[block_id];
  T* dst = arg.dst[block_id];
  const int64_t num_elem = arg.num_elem[block_id];
  static_assert(PACK_SIZE % sizeof(T) == 0, "");
  const int32_t num_elem_per_line = LINE_SIZE / sizeof(T);
  const int64_t num_line = num_elem / num_elem_per_line;
  const int32_t warp_id = thread_id / NUM_THREAD_PER_WARP;
  const int32_t lane_id = thread_id % NUM_THREAD_PER_WARP;
  ulong2 line_recv[NUM_PACK_PER_LINE_PER_THREAD];
  const int32_t offset = warp_id * NUM_PACK_PER_LINE_PER_WARP + lane_id;
  ulong2* dst_pack_ptr = reinterpret_cast<ulong2*>(dst) + offset;
  const ulong2* recv_pack_ptr = reinterpret_cast<const ulong2*>(recv) + offset;
  for (int64_t l = 0; l < num_line; ++l) {
#pragma unroll
    for (int32_t p = 0; p < NUM_PACK_PER_LINE_PER_THREAD; ++p) {
      FetchFunctor<Pack>()(line_recv[p], recv_pack_ptr + p * NUM_THREAD_PER_WARP);
    }
#pragma unroll
    for (int32_t p = 0; p < NUM_PACK_PER_LINE_PER_THREAD; ++p) {
      StoreFunctor<Pack>()(dst_pack_ptr + p * NUM_THREAD_PER_WARP, line_recv[p]);
    }
    dst_pack_ptr += NUM_PACK_PER_LINE;
    recv_pack_ptr += NUM_PACK_PER_LINE;
  }
}

}  // namespace

template<typename T>
void CudaRingAllReduceKernelUtil<T>::Send(DeviceCtx* ctx, CudaRingAllReduceArg<T> arg) {
  SendGpu<<<arg.num_rings, NUM_THREAD, 0, ctx->cuda_stream()>>>(arg);
}

template<typename T>
void CudaRingAllReduceKernelUtil<T>::RecvReduceSend(DeviceCtx* ctx, CudaRingAllReduceArg<T> arg) {
  RecvReduceSendGpu<<<arg.num_rings, NUM_THREAD, 0, ctx->cuda_stream()>>>(arg);
}

template<typename T>
void CudaRingAllReduceKernelUtil<T>::RecvReduceSendCopy(DeviceCtx* ctx,
                                                        CudaRingAllReduceArg<T> arg) {
  RecvReduceSendCopyGpu<<<arg.num_rings, NUM_THREAD, 0, ctx->cuda_stream()>>>(arg);
}

template<typename T>
void CudaRingAllReduceKernelUtil<T>::RecvSendCopy(DeviceCtx* ctx, CudaRingAllReduceArg<T> arg) {
  RecvSendCopyGpu<<<arg.num_rings, NUM_THREAD, 0, ctx->cuda_stream()>>>(arg);
}

template<typename T>
void CudaRingAllReduceKernelUtil<T>::RecvCopy(DeviceCtx* ctx, CudaRingAllReduceArg<T> arg) {
  RecvCopyGpu<<<arg.num_rings, NUM_THREAD, 0, ctx->cuda_stream()>>>(arg);
}

#define INSTANTIATE_CUDA_RING_ALL_REDUCE_KERNEL_UTIL(type_cpp, type_proto) \
  template struct CudaRingAllReduceKernelUtil<type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_CUDA_RING_ALL_REDUCE_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
