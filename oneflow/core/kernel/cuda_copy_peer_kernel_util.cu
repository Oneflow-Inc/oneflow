#include "oneflow/core/kernel/cuda_copy_peer_kernel_util.h"
#include "oneflow/core/kernel/kernel_util.cuh"

#define N_THREAD 1024
#define N_LOOP 16

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

__global__ void ReadKernel(void* buf, const void* src, volatile int32_t* step_mutex, size_t size) {
  const int32_t step_size = N_THREAD * N_LOOP * sizeof(ulong2);
  const int32_t n_step = DivUp(size, step_size);
  const int32_t thread_id = threadIdx.x;
  if (thread_id == 0) { assert(*step_mutex == 0); }
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
    __syncthreads();
    __threadfence_system();
    if (thread_id == 0) { *step_mutex = step + 1; }
  }
}

__global__ void WriteKernel(void* dst, const void* buf, volatile int32_t* step_mutex, size_t size) {
  const int32_t step_size = N_THREAD * N_LOOP * sizeof(ulong2);
  const int32_t n_step = DivUp(size, step_size);
  const int32_t thread_id = threadIdx.x;
  __syncthreads();
  for (int32_t step = 0; step < n_step; ++step) {
    if (thread_id == 0) {
      const int32_t next_step = step + 1;
      while (*step_mutex < next_step) {}
    }
    __syncthreads();
    __threadfence_system();
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
                                       size_t size, int32_t dst_dev_id, int32_t src_dev_id,
                                       cudaStream_t read, cudaStream_t write) {
  dim3 dim_grid(1, 1, 1);
  dim3 dim_block(N_THREAD, 1, 1);
  struct cudaLaunchParams params[2];
  void* read_kernel_args[] = {(void*)(&buf), (void*)(&src), (void*)(&step_mutex), (void*)(&size)};
  void* write_kernel_args[] = {(void*)(&dst), (void*)(&buf), (void*)(&step_mutex), (void*)(&size)};
  params[0].func = (void*)ReadKernel;
  params[0].gridDim = dim_grid;
  params[0].blockDim = dim_block;
  params[0].sharedMem = 0;
  params[0].args = read_kernel_args;
  params[0].stream = read;

  params[1].func = (void*)WriteKernel;
  params[1].gridDim = dim_grid;
  params[1].blockDim = dim_block;
  params[1].sharedMem = 0;
  params[1].args = write_kernel_args;
  params[1].stream = write;

  CudaCheck(cudaLaunchCooperativeKernelMultiDevice(
      params, 2,
      cudaCooperativeLaunchMultiDeviceNoPreSync | cudaCooperativeLaunchMultiDeviceNoPostSync));
}

}  // namespace oneflow
