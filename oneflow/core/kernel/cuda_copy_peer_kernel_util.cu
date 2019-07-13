#include "oneflow/core/kernel/cuda_copy_peer_kernel_util.h"
#include "oneflow/core/kernel/kernel_util.cuh"

constexpr int32_t PACK_SIZE = sizeof(ulong2);
constexpr int32_t PACK_ALIGN = alignof(ulong2);
constexpr int32_t NUM_THREAD = 256;
constexpr int32_t NUM_BLOCK_PER_CHUNK = 16;
constexpr int32_t BLOCK_SIZE = NUM_THREAD * PACK_SIZE;
constexpr int32_t CHUNK_SIZE = BLOCK_SIZE * NUM_BLOCK_PER_CHUNK;
constexpr int32_t DEFAULT_CHUNK_BUF_CAP = 2;

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

__forceinline__ __device__ void CopyChunk(void* dst, const void* src) {
#pragma unroll
  for (int32_t i = 0; i < NUM_BLOCK_PER_CHUNK; ++i) {
    FetchStore(dst, src);
    dst = (unsigned char*)(dst) + BLOCK_SIZE;
    src = (const unsigned char*)(src) + BLOCK_SIZE;
  }
}

__forceinline__ __device__ void CopyPartialChunk(void* dst, const void* src, int32_t size) {
  int32_t offset = 0;
  for (int32_t i = 0; i < NUM_BLOCK_PER_CHUNK; ++i) {
    if (offset < size) {
      FetchStore((unsigned char*)(dst) + offset, (const unsigned char*)(src) + offset);
    }
    offset += BLOCK_SIZE;
  }
}

__forceinline__ __device__ void Send(const void* src, const int32_t size, const int32_t thread_id,
                                     void* buf_ptr, const int32_t buf_cap,
                                     volatile int32_t* send_cnt_ptr,
                                     volatile int32_t* recv_cnt_ptr) {
  const int32_t num_step = DivUp(size, CHUNK_SIZE);
  int32_t remaining = size;
  for (int32_t step = 0; step < num_step; ++step) {
    if (thread_id == 0) {
      while (step - *recv_cnt_ptr < buf_cap) {}
    }
    __syncthreads();
    void* cur_buf_ptr = (unsigned char*)buf_ptr + (step % buf_cap) * CHUNK_SIZE;
    if (remaining >= CHUNK_SIZE) {
      CopyChunk(cur_buf_ptr, src);
    } else {
      CopyPartialChunk(cur_buf_ptr, src, remaining);
    }
    remaining -= CHUNK_SIZE;
    src = (const unsigned char*)(src) + CHUNK_SIZE;
    __threadfence_system();
    __syncthreads();
    if (thread_id == 0) { *send_cnt_ptr = step + 1; }
  }
}

__forceinline__ __device__ void Recv(void* dst, const int32_t size, const int32_t thread_id,
                                     const void* buf_ptr, const int32_t buf_cap,
                                     volatile int32_t* send_cnt_ptr,
                                     volatile int32_t* recv_cnt_ptr) {
  const int32_t num_step = DivUp(size, CHUNK_SIZE);
  int32_t remaining = size;
  for (int32_t step = 0; step < num_step; ++step) {
    if (thread_id == 0) {
      while (*send_cnt_ptr <= step) {}
    }
    __syncthreads();
    __threadfence_system();
    void* cur_buf_ptr = (unsigned char*)buf_ptr + (step % buf_cap) * CHUNK_SIZE;
    if (remaining >= CHUNK_SIZE) {
      CopyChunk(dst, cur_buf_ptr);
    } else {
      CopyPartialChunk(dst, cur_buf_ptr, remaining);
    }
    remaining -= CHUNK_SIZE;
    dst = (unsigned char*)(dst) + CHUNK_SIZE;
    __syncthreads();
    if (thread_id == 0) { *recv_cnt_ptr = step + 1; }
  }
  if (thread_id == 0) {
    *recv_cnt_ptr = 0;
    *send_cnt_ptr = 0;
  }
}

__global__ void Copy(void* dst, const void* src, const int32_t size, void* buf_ptr,
                     const int32_t buf_cap, volatile int32_t* send_cnt_ptr,
                     volatile int32_t* recv_cnt_ptr, bool send_or_recv) {
  const int32_t thread_id = threadIdx.x;
  if (send_or_recv) {
    Send(src, size, thread_id, buf_ptr, buf_cap, send_cnt_ptr, recv_cnt_ptr);
  } else {
    Recv(dst, size, thread_id, buf_ptr, buf_cap, send_cnt_ptr, recv_cnt_ptr);
  }
}

}  // namespace

struct CudaCopyPeerCtx {
  int32_t dst_dev_id;
  int32_t src_dev_id;
  cudaStream_t recv_stream;
  cudaStream_t send_stream;
  int32_t* recv_cnt_ptr;
  int32_t* send_cnt_ptr;
  void* buf_ptr;
  int32_t buf_cap;
};

void CudaCopyPeerKernelUtil::CtxCreate(CudaCopyPeerCtx** ctx, int32_t dst_dev_id,
                                       int32_t src_dev_id, cudaStream_t recv_stream) {
  *ctx = new CudaCopyPeerCtx();
  (*ctx)->dst_dev_id = dst_dev_id;
  (*ctx)->src_dev_id = src_dev_id;
  (*ctx)->recv_stream = recv_stream;
  WithCudaDevice(src_dev_id, [ctx]() { CudaCheck(cudaStreamCreate(&((*ctx)->send_stream))); });
  CudaCheck(cudaMallocHost(&((*ctx)->recv_cnt_ptr), sizeof(int32_t)));
  CudaCheck(cudaMallocHost(&((*ctx)->send_cnt_ptr), sizeof(int32_t)));
  *((*ctx)->recv_cnt_ptr) = 0;
  *((*ctx)->send_cnt_ptr) = 0;
  (*ctx)->buf_cap = DEFAULT_CHUNK_BUF_CAP;
  CudaCheck(cudaMallocHost(&((*ctx)->buf_ptr), CHUNK_SIZE * (*ctx)->buf_cap));
  CHECK_EQ(reinterpret_cast<std::uintptr_t>((*ctx)->buf_ptr) % PACK_ALIGN, 0);
}

void CudaCopyPeerKernelUtil::CtxDestroy(CudaCopyPeerCtx* ctx) {
  WithCudaDevice(ctx->src_dev_id, [ctx]() {
    CudaCheck(cudaStreamSynchronize(ctx->send_stream));
    CudaCheck(cudaStreamDestroy(ctx->send_stream));
  });
  CudaCheck(cudaFreeHost(ctx->recv_cnt_ptr));
  CudaCheck(cudaFreeHost(ctx->send_cnt_ptr));
  CudaCheck(cudaFreeHost(ctx->buf_ptr));
  delete ctx;
}

void CudaCopyPeerKernelUtil::CopyAsync(CudaCopyPeerCtx* ctx, void* dst, const void* src,
                                       int32_t size) {
  CHECK_EQ(size % PACK_SIZE, 0);
  CHECK_EQ(reinterpret_cast<std::uintptr_t>(dst) % PACK_ALIGN, 0);
  CHECK_EQ(reinterpret_cast<std::uintptr_t>(src) % PACK_ALIGN, 0);
  const bool launch_flag_send = true;
  const bool launch_flag_recv = false;
  cudaLaunchParams params[2];
  params[0].func = params[1].func = (void*)Copy;
  params[0].gridDim = params[1].gridDim = {1, 1, 1};
  params[0].blockDim = params[1].blockDim = {NUM_THREAD, 1, 1};
  params[0].sharedMem = params[1].sharedMem = 0;
  void* send_args[] = {(void*)(&dst),
                       (void*)(&src),
                       (void*)(&size),
                       (void*)(&(ctx->buf_ptr)),
                       (void*)(&(ctx->buf_cap)),
                       (void*)(&(ctx->send_cnt_ptr)),
                       (void*)(&(ctx->recv_cnt_ptr)),
                       (void*)(&launch_flag_send)};
  void* recv_args[] = {(void*)(&dst),
                       (void*)(&src),
                       (void*)(&size),
                       (void*)(&(ctx->buf_ptr)),
                       (void*)(&(ctx->buf_cap)),
                       (void*)(&(ctx->send_cnt_ptr)),
                       (void*)(&(ctx->recv_cnt_ptr)),
                       (void*)(&launch_flag_recv)};
  params[0].args = send_args;
  params[1].args = recv_args;
  params[0].stream = ctx->send_stream;
  params[1].stream = ctx->recv_stream;
  CudaCheck(cudaLaunchCooperativeKernelMultiDevice(params, 2));
}

}  // namespace oneflow
