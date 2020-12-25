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

namespace oneflow {

template<typename T>
OF_DEVICE_FUNC T MaxWithLogThreshold(T x) {
  const T threshold = 1e-20;
  return x > threshold ? x : threshold;
}

#if defined(__CUDACC__)
__device__ __forceinline__ half MaxWithLogThreshold(half x) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  half threshold = hexp2(__float2half(-14.0));
  if (__hgt(x, threshold)) { return x; }
  return threshold;
#else
  printf("use half need nvcc arch >= 530");
  assert(false);
#endif /* __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)*/
}
#endif

template<typename T>
OF_DEVICE_FUNC T SafeLog(T x) {
  return logf(MaxWithLogThreshold(x));
}

#if defined(__CUDACC__)
__device__ __forceinline__ half SafeLog(half x) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  return hlog(MaxWithLogThreshold(x));
#else
  printf("use half need nvcc arch >= 530");
  assert(false);
#endif /* __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)*/
}
#endif

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_KERNEL_UTIL_CUH_
