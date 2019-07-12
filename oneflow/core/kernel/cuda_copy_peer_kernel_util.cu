#include "oneflow/core/kernel/cuda_copy_peer_kernel_util.h"
#include "oneflow/core/kernel/kernel_util.cuh"

constexpr int32_t NUM_THREAD = 256;
constexpr int32_t NUM_BLOCK_PER_CHUNK = 16;
constexpr int32_t BLOCK_SIZE = NUM_THREAD * sizeof(ulong2);
constexpr int32_t CHUNK_SIZE = BLOCK_SIZE * NUM_BLOCK_PER_CHUNK;

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
  for (int32_t l = 0; l < NUM_BLOCK_PER_CHUNK; ++l) {
    FetchStore(dst, src);
    dst = (unsigned char*)(dst) + BLOCK_SIZE;
    src = (const unsigned char*)(src) + BLOCK_SIZE;
  }
}

__forceinline__ __device__ void CopyPartialChunk(void* dst, const void* src, int32_t size) {
  int32_t offset = 0;
  for (int32_t l = 0; l < NUM_BLOCK_PER_CHUNK; ++l) {
    if (offset < size) {
      FetchStore((unsigned char*)(dst) + offset, (const unsigned char*)(src) + offset);
    }
    offset += BLOCK_SIZE;
  }
}

__forceinline__ __device__ void Send(const void* src, size_t size, int32_t thread_id, void* buf_0,
                                     void* buf_1, volatile int32_t* send_step,
                                     volatile int32_t* recv_step) {
  const int32_t num_chunk = DivUp(size, CHUNK_SIZE);
  int32_t remaining = size;
  if (thread_id == 0) {
    assert(*send_step == 0);
    assert(*recv_step == 0);
  }
  __syncthreads();
  for (int32_t step = 0; step < num_chunk; ++step) {
    if (thread_id == 0) {
      while (step - *recv_step < 2) {}
    }
    __syncthreads();
    void* buf = step % 2 == 0 ? buf_0 : buf_1;
    if (remaining >= CHUNK_SIZE) {
      CopyChunk(buf, src);
    } else {
      CopyPartialChunk(buf, src, remaining);
    }
    remaining -= CHUNK_SIZE;
    src = (const unsigned char*)(src) + CHUNK_SIZE;
    __syncthreads();
    __threadfence_system();
    if (thread_id == 0) { *send_step = step + 1; }
  }
}

__forceinline__ __device__ void Recv(void* dst, size_t size, int32_t thread_id, const void* buf_0,
                                     const void* buf_1, volatile int32_t* send_step,
                                     volatile int32_t* recv_step) {
  const int32_t num_chunk = DivUp(size, CHUNK_SIZE);
  int32_t remaining = size;
  for (int32_t step = 0; step < num_chunk; ++step) {
    if (thread_id == 0) {
      while (*send_step <= step) {}
    }
    __syncthreads();
    __threadfence_system();
    const void* buf = step % 2 == 0 ? buf_0 : buf_1;
    if (remaining >= CHUNK_SIZE) {
      CopyChunk(dst, buf);
    } else {
      CopyPartialChunk(dst, buf, remaining);
    }
    remaining -= CHUNK_SIZE;
    dst = (unsigned char*)(dst) + CHUNK_SIZE;
    __syncthreads();
    if (thread_id == 0) { *recv_step = step + 1; }
  }
}

}  // namespace

void CudaCopyPeerKernelUtil::CopyAsync(void* dst, void* buf, const void* src, int32_t* step_mutex,
                                       size_t size, int32_t dst_dev_id, int32_t src_dev_id,
                                       cudaStream_t read, cudaStream_t write) {}

}  // namespace oneflow
