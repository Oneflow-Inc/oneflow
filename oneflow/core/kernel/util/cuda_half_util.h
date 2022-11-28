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
#ifndef ONEFLOW_CORE_KERNEL_UTIL_CUDA_HALF_UTIL_H_
#define ONEFLOW_CORE_KERNEL_UTIL_CUDA_HALF_UTIL_H_

#include "oneflow/core/device/cuda_util.h"

namespace oneflow {

#define HALF_CHECK_FAILED                                             \
  printf("half operations are only supported when CUDA_ARCH >= 530"); \
  assert(false)

__inline__ __device__ half hone() { return __float2half(1.0); }
__inline__ __device__ half hzero() { return __float2half(0.0); }

__inline__ half float16_2half(float16 x) {
  // TODO: Potential loss of accuracy
  half* ret = reinterpret_cast<half*>(&x);
  return *ret;
}

__inline__ float16 half2float16(half x) {
  // TODO: Potential loss of accuracy
  float16* ret = reinterpret_cast<float16*>(&x);
  return *ret;
}
}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_UTIL_CUDA_HALF_UTIL_H_
