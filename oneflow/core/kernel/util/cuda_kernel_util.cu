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
#include "oneflow/core/kernel/util/cuda_kernel_util.h"
#include "oneflow/core/common/data_type.h"

namespace oneflow {

template<typename T>
__device__ T MaxWithLogThreshold(T x) {
  const T threshold = 1e-20;
  return x > threshold ? x : threshold;
}

template __device__ float MaxWithLogThreshold(float x);
template __device__ double MaxWithLogThreshold(double x);

template<>
__device__ half MaxWithLogThreshold(half x) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  half threshold = hexp2(__float2half(-14.0));
  if (__hgt(x, threshold)) { return x; }
  return threshold;
#else
  printf("use half need nvcc arch >= 530");
  assert(false);
#endif /* __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)*/
}

template<typename T>
__device__ T SafeLog(T x) {
  return logf(MaxWithLogThreshold(x));
}

template __device__ float SafeLog(float x);
template __device__ double SafeLog(double x);

template<>
__device__ half SafeLog(half x) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  return hlog(MaxWithLogThreshold<half>(x));
#else
  printf("use half need nvcc arch >= 530");
  assert(false);
#endif /* __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)*/
}

}  // namespace oneflow
