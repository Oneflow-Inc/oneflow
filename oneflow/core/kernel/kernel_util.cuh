#ifndef ONEFLOW_CORE_KERNEL_KERNEL_UTIL_CUH_
#define ONEFLOW_CORE_KERNEL_KERNEL_UTIL_CUH_

namespace oneflow {

template<typename T>
inline __device__ T gpu_atomic_add(T* address, const T val);

template<>
inline __device__ float gpu_atomic_add(float* address, const float val) {
  return atomicAdd(address, val);
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_KERNEL_UTIL_CUH_
