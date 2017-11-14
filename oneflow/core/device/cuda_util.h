#ifndef ONEFLOW_CORE_DEVICE_CUDA_UTIL_H_
#define ONEFLOW_CORE_DEVICE_CUDA_UTIL_H_

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <curand.h>
#include "oneflow/core/common/util.h"

namespace oneflow {

#ifdef USE_CUDNN
namespace cudnn {

template<typename T>
class DataType;

template<>
class DataType<float> {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_FLOAT;
  static float oneval, zeroval;
  static const void* one;
  static const void* zero;
};

template<>
class DataType<double> {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_DOUBLE;
  static double oneval, zeroval;
  static const void* one;
  static const void* zero;
};

/*
template<>
class DataType<signed char> {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_INT8;
  static signed char oneval, zeroval;
  static const void* one;
  static const void* zero;
};

template<>
class DataType<int> {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_INT32;
  static int oneval, zeroval;
  static const void* one;
  static const void* zero;
};
*/

}  // namespace cudnn
#endif  // USE_CUDNN

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

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DEVICE_CUDA_UTIL_H_
