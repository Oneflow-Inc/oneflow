#include "oneflow/core/kernel/cuda_copy_peer_kernel_util.h"
#include "oneflow/core/kernel/kernel_util.cuh"

#define N_THREAD 256
#define N_LOOP 16

namespace oneflow {

namespace {

inline __device__ void Fetch(ulong2& v, const ulong2* p) {
  // clang-format off
  asm volatile("ld.volatile.global.v2.u64 {%0,%1}, [%2];" : "=l"(v.x), "=l"(v.y) : "l"(p) : "memory");
  // clang-format on
}
inline __device__ void Store(ulong2* p, ulong2& v) {
  // clang-format off
  asm volatile("st.volatile.global.v2.u64 [%0], {%1,%2};" :: "l"(p), "l"(v.x), "l"(v.y) : "memory");
  // clang-format on
}

__global__ void ReadKernel(void* buf, const void* src, volatile int32_t* step_mutex, size_t size) {
  const int32_t step_size = N_THREAD * N_LOOP * sizeof(ulong2);
  const int32_t n_step = RoundUp(size, step_size) / step_size;
  const int32_t thread_id = threadIdx.x;
  if (thread_id == 0) {
    int32_t old = 0;
    do { old = atomicCAS(step_mutex, old, 0); } while (old != 0);
  }
  __syncthreads();
  for (int32_t step = 0; step < n_step; ++step) {
    int32_t step_offset = step * step_size;
    ulong2 v;
#pragma unroll
    for (int32_t l = 0; l < N_LOOP; ++l) {
      const int32_t offset = step_offset + (l * N_THREAD + thread_id) * sizeof(ulong2);
      if (offset < size) {
        Fetch(v, reinterpret_cast<const ulong2*>(static_cast<const uint8_t*>(src) + offset));
        Store(reinterpret_cast<ulong2*>(static_cast<uint8_t*>(buf) + offset), v);
      }
    }
    __threadfence_system();
    assert(atomicCAS(step_mutex, step, step + 1) == step);
  }
}

__global__ void WriteKernel(void* dst, const void* buf, int32_t* step_mutex, size_t size) {
  const int32_t step_size = N_THREAD * N_LOOP * sizeof(ulong2);
  const size_t n_step = RoundUp(size, step_size) / step_size;
  const int32_t thread_id = threadIdx.x;
  if (thread_id == 0) {
    do { } while (atomicCAS(step_mutex, 0, 0) != 0); }
  __syncthreads();
  for (int32_t step = 0; step < n_step; ++step) {
    if (thread_id == 0) {
      const int32_t next_step = step + 1;
      do { } while (atomicCAS(step_mutex, next_step, next_step) != next_step); }
    __syncthreads();
    int32_t step_offset = step * step_size;
    ulong2 v;
#pragma unroll
    for (int32_t l = 0; l < N_LOOP; ++l) {
      const int32_t offset = step_offset + (l * N_THREAD + thread_id) * sizeof(ulong2);
      if (offset < size) {
        Fetch(v, reinterpret_cast<const ulong2*>(static_cast<const uint8_t*>(buf) + offset));
        Store(reinterpret_cast<ulong2*>(static_cast<uint8_t*>(dst) + offset), v);
      }
    }
  }
}

}  // namespace

void CudaCopyPeerKernelUtil::CopyAsync(void* dst, void* buf, const void* src, int32_t* step_mutex,
                                       size_t size, cudaStream_t read, cudaStream_t write) {
  ReadKernel<<<1, N_THREAD, 0, read>>>(buf, src, step_mutex, size);
  WriteKernel<<<1, N_THREAD, 0, write>>>(dst, buf, step_mutex, size)
}

}  // namespace oneflow
