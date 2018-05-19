#ifndef ONEFLOW_CORE_KERNEL_KERNEL_UTIL_CUH_
#define ONEFLOW_CORE_KERNEL_KERNEL_UTIL_CUH_

namespace oneflow {

template<typename T>
__device__ T gpu_atomic_add(T* address, const T val);

template<typename T>
__device__ T gpu_atomic_max(T* address, const T val);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_KERNEL_UTIL_CUH_
