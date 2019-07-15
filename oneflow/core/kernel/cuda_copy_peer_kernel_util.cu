#include "oneflow/core/kernel/cuda_copy_peer_kernel_util.h"
#include "oneflow/core/kernel/kernel_util.cuh"

constexpr int32_t PACK_SIZE = sizeof(ulong2);
constexpr int32_t PACK_ALIGN = alignof(ulong2);
constexpr int32_t NUM_THREAD = 1024;
constexpr int32_t NUM_STEP_PER_CHUNK = 16;
constexpr int32_t STEP_SIZE = NUM_THREAD * PACK_SIZE;
constexpr int32_t CHUNK_SIZE = STEP_SIZE * NUM_STEP_PER_CHUNK;
constexpr int32_t DEFAULT_CHUNK_BUF_CAP = 2;
constexpr int32_t WARP_SIZE = 32;
constexpr int32_t MAX_NUM_BLOCK = 2;

namespace oneflow {

namespace {

__forceinline__ __device__ int32_t DivUp(int32_t n, int32_t val) { return (n + val - 1) / val; }

__forceinline__ __device__ void Fetch(ulong2& v, const ulong2* p) {
  // clang-format off
  asm volatile("ld.volatile.global.v2.u64 {%0,%1}, [%2];" : "=l"(v.x), "=l"(v.y) : "l"(p) : "memory");
  // clang-format on
}

__forceinline__ __device__ void Store(ulong2* p, ulong2& v) {
  // clang-format off
  asm volatile("st.volatile.global.v2.u64 [%0], {%1,%2};" :: "l"(p), "l"(v.x), "l"(v.y) : "memory");
  // clang-format on
}

__forceinline__ __device__ void FetchStore(ulong2* dst, const ulong2* src) {
  ulong2 v;
  Fetch(v, src);
  Store(dst, v);
}

__forceinline__ __device__ void FetchStore(void* dst, const void* src) {
  FetchStore(reinterpret_cast<ulong2*>(dst), reinterpret_cast<const ulong2*>(src));
}

__forceinline__ __device__ void CopyChunk(void* dst, const void* src, const int32_t thread_id) {
  dst = (unsigned char*)(dst) + thread_id * PACK_SIZE;
  src = (unsigned char*)(src) + thread_id * PACK_SIZE;
#pragma unroll
  for (int32_t i = 0; i < NUM_STEP_PER_CHUNK; ++i) {
    FetchStore(dst, src);
    dst = (unsigned char*)(dst) + STEP_SIZE;
    src = (const unsigned char*)(src) + STEP_SIZE;
  }
}

__forceinline__ __device__ void CopyPartialChunk(void* dst, const void* src, const int32_t size,
                                                 const int32_t thread_id) {
  int32_t offset = thread_id * PACK_SIZE;
  for (int32_t i = 0; i < NUM_STEP_PER_CHUNK; ++i) {
    if (offset < size) {
      FetchStore((unsigned char*)(dst) + offset, (const unsigned char*)(src) + offset);
    }
    offset += STEP_SIZE;
  }
}

__forceinline__ __device__ void Send(const void* src, const int32_t size, const int32_t thread_id,
                                     void* buf_ptr, const int32_t buf_cap,
                                     volatile int32_t* send_cnt_ptr,
                                     volatile int32_t* recv_cnt_ptr) {
  const int32_t num_chunk = DivUp(size, CHUNK_SIZE);
  int32_t remaining = size;
  if (thread_id == 0) {
    while (*recv_cnt_ptr != 0) {}
    while (*send_cnt_ptr != 0) {}
  }
  __syncthreads();
  for (int32_t chunk = 0; chunk < num_chunk; ++chunk) {
    if (thread_id == WARP_SIZE) {
      while (chunk - *recv_cnt_ptr >= buf_cap) {}
    }
    __syncthreads();
    void* cur_buf_ptr = (unsigned char*)buf_ptr + (chunk % buf_cap) * CHUNK_SIZE;
    if (remaining >= CHUNK_SIZE) {
      CopyChunk(cur_buf_ptr, src, thread_id);
    } else {
      CopyPartialChunk(cur_buf_ptr, src, remaining, thread_id);
    }
    remaining -= CHUNK_SIZE;
    src = (const unsigned char*)(src) + CHUNK_SIZE;
    __threadfence_system();
    __syncthreads();
    if (thread_id == 0) { *send_cnt_ptr = chunk + 1; }
  }
}

__forceinline__ __device__ void Recv(void* dst, const int32_t size, const int32_t thread_id,
                                     const void* buf_ptr, const int32_t buf_cap,
                                     volatile int32_t* send_cnt_ptr,
                                     volatile int32_t* recv_cnt_ptr) {
  const int32_t num_chunk = DivUp(size, CHUNK_SIZE);
  int32_t remaining = size;
  for (int32_t chunk = 0; chunk < num_chunk; ++chunk) {
    if (thread_id == WARP_SIZE) {
      while (*send_cnt_ptr <= chunk) {}
    }
    __syncthreads();
    void* cur_buf_ptr = (unsigned char*)buf_ptr + (chunk % buf_cap) * CHUNK_SIZE;
    if (remaining >= CHUNK_SIZE) {
      CopyChunk(dst, cur_buf_ptr, thread_id);
    } else {
      CopyPartialChunk(dst, cur_buf_ptr, remaining, thread_id);
    }
    remaining -= CHUNK_SIZE;
    dst = (unsigned char*)(dst) + CHUNK_SIZE;
    __syncthreads();
    if (thread_id == 0) { *recv_cnt_ptr = chunk + 1; }
  }
  if (thread_id == 0) {
    *recv_cnt_ptr = 0;
    *send_cnt_ptr = 0;
  }
}

__launch_bounds__(NUM_THREAD) __global__
    void Copy(void* dst, const void* src, const int32_t size, void* buf_ptr, const int32_t buf_cap,
              int32_t* send_cnt_ptr, int32_t* recv_cnt_ptr, bool send_or_recv) {
  const int32_t block_id = blockIdx.x;
  const int32_t num_block = gridDim.x;
  const int32_t thread_id = threadIdx.x;
  const int32_t block_size = DivUp(size / PACK_SIZE, num_block) * PACK_SIZE;
  void* this_block_dst = reinterpret_cast<unsigned char*>(dst) + block_size * block_id;
  const void* this_block_src = reinterpret_cast<const unsigned char*>(src) + block_size * block_id;
  const int32_t this_block_size =
      (block_id + 1) * block_size <= size ? block_size : max(0, size - block_id * block_size);
  void* this_buf_ptr = reinterpret_cast<unsigned char*>(buf_ptr) + CHUNK_SIZE * buf_cap * block_id;
  int32_t* this_send_cnt_ptr = send_cnt_ptr + block_id;
  int32_t* this_recv_cnt_ptr = recv_cnt_ptr + block_id;
  if (send_or_recv) {
    Send(this_block_src, this_block_size, thread_id, this_buf_ptr, buf_cap, this_send_cnt_ptr,
         this_recv_cnt_ptr);
  } else {
    Recv(this_block_dst, this_block_size, thread_id, this_buf_ptr, buf_cap, this_send_cnt_ptr,
         this_recv_cnt_ptr);
  }
}

__global__ void Null() {}

}  // namespace

struct CudaCopyPeerCtx {
  int32_t dst_dev_id;
  int32_t src_dev_id;
  cudaStream_t recv_stream;
  cudaStream_t send_stream;
  int32_t num_block;
  int32_t* recv_cnt_ptr;
  int32_t* send_cnt_ptr;
  void* buf_ptr;
  int32_t buf_cap;
  bool p2p_enabled;
};

void CudaCopyPeerKernelUtil::CtxCreate(CudaCopyPeerCtx** ctx, int32_t dst_dev_id,
                                       int32_t src_dev_id, cudaStream_t recv_stream) {
  *ctx = new CudaCopyPeerCtx();
  (*ctx)->dst_dev_id = dst_dev_id;
  (*ctx)->src_dev_id = src_dev_id;
  (*ctx)->recv_stream = recv_stream;

  WithCudaDevice(dst_dev_id, [ctx]() {
    int32_t can_access;
    CudaCheck(cudaDeviceCanAccessPeer(&can_access, (*ctx)->dst_dev_id, (*ctx)->src_dev_id));
    if (can_access) {
      cudaError_t error = cudaDeviceEnablePeerAccess((*ctx)->src_dev_id, 0);
      if (error != cudaErrorPeerAccessAlreadyEnabled) { CudaCheck(error); }
      (*ctx)->p2p_enabled = true;
    } else {
      (*ctx)->p2p_enabled = false;
    }
  });
  if (!(*ctx)->p2p_enabled) {
    WithCudaDevice(src_dev_id, [ctx]() { CudaCheck(cudaStreamCreate(&((*ctx)->send_stream))); });
    (*ctx)->num_block = MAX_NUM_BLOCK;
    NumaAwareCudaMallocHost((*ctx)->dst_dev_id, reinterpret_cast<void**>(&((*ctx)->recv_cnt_ptr)),
                            sizeof(int32_t) * (*ctx)->num_block);
    NumaAwareCudaMallocHost((*ctx)->dst_dev_id, reinterpret_cast<void**>(&((*ctx)->send_cnt_ptr)),
                            sizeof(int32_t) * (*ctx)->num_block);
    *((*ctx)->recv_cnt_ptr) = 0;
    *((*ctx)->send_cnt_ptr) = 0;
    (*ctx)->buf_cap = DEFAULT_CHUNK_BUF_CAP;
    NumaAwareCudaMallocHost((*ctx)->dst_dev_id, reinterpret_cast<void**>(&((*ctx)->buf_ptr)),
                            CHUNK_SIZE * (*ctx)->buf_cap * (*ctx)->num_block);
    CHECK_EQ(reinterpret_cast<std::uintptr_t>((*ctx)->buf_ptr) % PACK_ALIGN, 0);
  }
}

void CudaCopyPeerKernelUtil::CtxDestroy(CudaCopyPeerCtx* ctx) {
  if (!ctx->p2p_enabled) {
    WithCudaDevice(ctx->src_dev_id, [ctx]() {
      CudaCheck(cudaStreamSynchronize(ctx->send_stream));
      CudaCheck(cudaStreamDestroy(ctx->send_stream));
    });
    CudaCheck(cudaFreeHost(ctx->recv_cnt_ptr));
    CudaCheck(cudaFreeHost(ctx->send_cnt_ptr));
    CudaCheck(cudaFreeHost(ctx->buf_ptr));
  }
  delete ctx;
}

void CudaCopyPeerKernelUtil::CopyAsync(CudaCopyPeerCtx* ctx, void* dst, const void* src,
                                       int32_t size) {
  if (ctx->p2p_enabled) {
    CHECK_EQ(size % PACK_SIZE, 0);
    CHECK_EQ(reinterpret_cast<std::uintptr_t>(dst) % PACK_ALIGN, 0);
    CHECK_EQ(reinterpret_cast<std::uintptr_t>(src) % PACK_ALIGN, 0);
  } else {
    CHECK_EQ(size % PACK_SIZE, 0);
    CHECK_EQ(reinterpret_cast<std::uintptr_t>(dst) % PACK_ALIGN, 0);
    CHECK_EQ(reinterpret_cast<std::uintptr_t>(src) % PACK_ALIGN, 0);
    WithCudaDevice(ctx->src_dev_id, [&]() {
      Copy<<<ctx->num_block, NUM_THREAD, 0, ctx->send_stream>>>(
          dst, src, size, ctx->buf_ptr, ctx->buf_cap, ctx->send_cnt_ptr, ctx->recv_cnt_ptr, true);
    });
    WithCudaDevice(ctx->dst_dev_id, [&]() {
      Copy<<<ctx->num_block, NUM_THREAD, 0, ctx->recv_stream>>>(
          dst, src, size, ctx->buf_ptr, ctx->buf_cap, ctx->send_cnt_ptr, ctx->recv_cnt_ptr, false);
    });
    cudaLaunchParams params[2];
    params[0].func = params[1].func = (void*)Null;
    params[0].gridDim = params[1].gridDim = {1, 1, 1};
    params[0].blockDim = params[1].blockDim = {1, 1, 1};
    params[0].sharedMem = params[1].sharedMem = 0;
    void* args[] = {};
    params[0].args = params[1].args = args;
    params[0].stream = ctx->send_stream;
    params[1].stream = ctx->recv_stream;
    CudaCheck(cudaLaunchCooperativeKernelMultiDevice(params, 2));
  }
}

}  // namespace oneflow
