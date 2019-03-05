#ifndef ONEFLOW_CORE_KERNEL_KERNEL_UTIL_CUH_
#define ONEFLOW_CORE_KERNEL_KERNEL_UTIL_CUH_

#include <cuda_fp16.h>

namespace oneflow {

template<typename T>
__device__ T gpu_atomic_add(T* address, const T val);

template<typename T>
__device__ T gpu_atomic_max(T* address, const T val);

template<typename T>
__host__ __device__ T MaxWithLogThreshold(T x) {
  const T threshold = 1e-20;
  return x > threshold ? x : threshold;
}

template<>
__host__ __device__ half MaxWithLogThreshold(half x) {
  const half threshold = hexp2(-14.0_h);
  if(__hgt(x, threshold)) {
    return x;
  }
  return threshold;
}

template<typename T>
__host__ __device__ T SafeLog(T x) {
  return logf(MaxWithLogThreshold(x));
}

template<>
__host__ __device__ half SafeLog(half x) {
  return hlog(MaxWithLogThreshold(x));
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_KERNEL_UTIL_CUH_
