#ifndef ONEFLOW_CORE_KERNEL_UTIL_CUDA_KERNEL_UTIL_H_
#define ONEFLOW_CORE_KERNEL_UTIL_CUDA_KERNEL_UTIL_H_

namespace oneflow {

template<typename T>
__device__ T gpu_atomic_add(T* address, const T val);

template<typename T>
__device__ T gpu_atomic_max(T* address, const T val);

template<typename T>
__device__ T MaxWithLogThreshold(T x);

template<typename T>
__device__ T SafeLog(T x);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_UTIL_CUDA_KERNEL_UTIL_H_
