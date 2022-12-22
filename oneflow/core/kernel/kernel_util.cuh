/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_CORE_KERNEL_KERNEL_UTIL_CUH_
#define ONEFLOW_CORE_KERNEL_KERNEL_UTIL_CUH_
#include "oneflow/core/cuda/atomic.cuh"
#include "oneflow/core/device/cuda_pseudo_half.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/device/cuda_util.h"

namespace oneflow {

template<typename T, typename std::enable_if<IsFloating<T>::value>::type* = nullptr>
OF_DEVICE_FUNC T MaxWithLogThreshold(T x) {
  const T threshold = 1e-20;
  return x > threshold ? x : threshold;
}

template<typename T, typename std::enable_if<IsIntegral<T>::value>::type* = nullptr>
OF_DEVICE_FUNC T MaxWithLogThreshold(T x) {
  return x;
}

#if defined(__CUDACC__)
__device__ __forceinline__ half MaxWithLogThreshold(half x) {
  half threshold = hexp2(__float2half(-14.0));
  if (__hgt(x, threshold)) { return x; }
  return threshold;
}
#endif

template<typename T>
OF_DEVICE_FUNC T SafeLog(T x) {
  return logf(MaxWithLogThreshold(x));
}

#if defined(__CUDACC__)
__device__ __forceinline__ half SafeLog(half x) { return hlog(MaxWithLogThreshold(x)); }
#endif

#if defined(__CUDACC__)
template<typename T, typename IndexT,
         typename std::enable_if<std::is_same<half, T>::value>::type* = nullptr>
__device__ __forceinline__ void fastSpecializedAtomicAdd(T* tensor, IndexT index,
                                                         const IndexT numel, T value) {
#if ((defined(CUDA_VERSION) && (CUDA_VERSION < 10000)) \
     || (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)))
  cuda::atomic::Add(reinterpret_cast<half*>(tensor) + index, static_cast<half>(value));
#else
  // Accounts for the chance tensor falls on an odd 16 bit alignment (ie, not 32 bit aligned)
  __half* target_addr = reinterpret_cast<__half*>(tensor + index);
  bool low_byte = (reinterpret_cast<std::uintptr_t>(target_addr) % sizeof(__half2) == 0);

  if (low_byte && index < (numel - 1)) {
    __half2 value2;
    value2.x = value;
    value2.y = __float2half_rz(0);
    cuda::atomic::Add(reinterpret_cast<__half2*>(target_addr), value2);

  } else if (!low_byte && index > 0) {
    __half2 value2;
    value2.x = __float2half_rz(0);
    value2.y = value;
    cuda::atomic::Add(reinterpret_cast<__half2*>(target_addr - 1), value2);

  } else {
    cuda::atomic::Add(reinterpret_cast<__half*>(tensor) + index, static_cast<__half>(value));
  }
#endif
}

template<typename T, typename IndexT,
         typename std::enable_if<!std::is_same<half, T>::value>::type* = nullptr>
__device__ __forceinline__ void fastSpecializedAtomicAdd(T* tensor, IndexT index,
                                                         const IndexT numel, T value) {
  cuda::atomic::Add(tensor + index, value);
}

template<class T, class IndexT>
__device__ __forceinline__ void fastAtomicAdd(T* tensor, IndexT index, const IndexT numel, T value,
                                              bool fast_atomics) {
  if (fast_atomics) {
    fastSpecializedAtomicAdd(tensor, index, numel, value);
  } else {
    atomicAdd(tensor + index, value);
  }
}
#endif

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_KERNEL_UTIL_CUH_
