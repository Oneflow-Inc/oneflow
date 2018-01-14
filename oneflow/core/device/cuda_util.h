#ifndef ONEFLOW_CORE_DEVICE_CUDA_UTIL_H_
#define ONEFLOW_CORE_DEVICE_CUDA_UTIL_H_

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <curand.h>
#include "oneflow/core/common/util.h"

namespace oneflow {

template<typename T>
void CudaCheck(T error);

// CUDA: grid stride looping
#define CUDA_1D_KERNEL_LOOP(i, n)                                  \
  for (int32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

const int32_t kCudaThreadsNumPerBlock = 512;
const int32_t kCudaMaxBlocksNum = 4096;

inline int32_t BlocksNum4ThreadsNum(const int32_t n) {
  return std::min((n + kCudaThreadsNumPerBlock - 1) / kCudaThreadsNumPerBlock,
                  kCudaMaxBlocksNum);
}

size_t GetAvailableGpuMemSize(int dev_id);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DEVICE_CUDA_UTIL_H_
