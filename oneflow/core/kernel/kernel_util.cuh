#ifndef ONEFLOW_CORE_KERNEL_KERNEL_UTIL_CUH_
#define ONEFLOW_CORE_KERNEL_KERNEL_UTIL_CUH_
#include <cub/cub.cuh>

namespace oneflow {

template<typename T>
inline __device__ T gpu_atomic_add(T* address, const T val);

template<>
inline __device__ float gpu_atomic_add(float* address, const float val) {
  return atomicAdd(address, val);
}

template<>
inline __device__ double gpu_atomic_add(double* address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_KERNEL_UTIL_CUH_
