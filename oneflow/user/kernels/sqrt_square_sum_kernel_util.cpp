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
#include "oneflow/user/kernels/sqrt_square_sum_kernel_util.h"

namespace oneflow {

template<typename T>
struct SqrtSquareSumKernelUtil<DeviceType::kCPU, T> {
  static void SqrtSquareSum(ep::Stream* stream, int64_t n, const T* x, T* y, T* tmp) {
    T sum = 0;
    FOR_RANGE(int64_t, i, 0, n) { sum += x[i] * x[i]; }
    *y = std::sqrt(sum);
  }
};

#define INSTANTIATE_SQUARE_SUM_KERNEL_UTIL_CPU(type_cpp, type_proto) \
  template struct SqrtSquareSumKernelUtil<DeviceType::kCPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_SQUARE_SUM_KERNEL_UTIL_CPU, FLOATING_DATA_TYPE_SEQ);
#undef INSTANTIATE_SQUARE_SUM_KERNEL_UTIL_CPU

}  // namespace oneflow
