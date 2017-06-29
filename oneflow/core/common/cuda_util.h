#ifndef ONEFLOW_CORE_COMMON_CUDA_UTIL_H_
#define ONEFLOW_CORE_COMMON_CUDA_UTIL_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

inline void CudaCheck(cudaError_t error) {
  CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error);
}

// CUDA: grid stride looping
#define CUDA_1D_KERNEL_LOOP(i, n) \
  for (int32_t i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

// CUDA: check for error after kernel execution and exit loudly if there is one.
inline void CudaPostKernelCheck() {
  CudaCheck(cudaPeekAtLastError());
}

const int32_t kCudaThreadsNumPerBlock = 512;
const int32_t kCudaMaxBlocksNum = 4096;

inline int32_t BlocksNum4ThreadsNum(const int32_t N) {
  return std::min((N + kCudaThreadsNumPerBlock - 1) / kCudaThreadsNumPerBlock, 
                  kCudaMaxBlocksNum);
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_CUDA_UTIL_H_
