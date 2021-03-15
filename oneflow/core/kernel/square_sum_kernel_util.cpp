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
#include "oneflow/core/kernel/square_sum_kernel_util.h"

namespace oneflow {

template<typename T>
struct SquareSumKernelUtil<DeviceType::kCPU, T> {
  static void SquareSum(DeviceCtx* ctx, int64_t n, const T* x, T* y) {
    T sum = 0;
    FOR_RANGE(int64_t, i, 0, n) { sum += x[i] * x[i]; }
    *y = sum;
  }

  static void MultiSquareSum(DeviceCtx* ctx, const std::vector<SquareSumParam<T>>& params, T* y) {
    T sum = 0;
    FOR_RANGE(int64_t, i, 0, params.size()) {
      const auto& p = params[i];
      FOR_RANGE(int64_t, j, 0, p.count) { sum += p.ptr[j] * p.ptr[j]; }
    }
    *y = sum;
  }
};

#define INSTANTIATE_SQUARE_SUM_KERNEL_UTIL_CPU(type_cpp, type_proto) \
  template struct SquareSumKernelUtil<DeviceType::kCPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_SQUARE_SUM_KERNEL_UTIL_CPU, FLOATING_DATA_TYPE_SEQ);
#undef INSTANTIATE_SQUARE_SUM_KERNEL_UTIL_CPU

}  // namespace oneflow
