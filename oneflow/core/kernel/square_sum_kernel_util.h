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
#ifndef ONEFLOW_CORE_KERNEL_SQUARE_SUM_KERNEL_UTIL_H_
#define ONEFLOW_CORE_KERNEL_SQUARE_SUM_KERNEL_UTIL_H_

#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<typename T>
struct SquareSumParam {
  const T* ptr;
  int64_t count;
};

template<DeviceType device_type, typename T>
struct SquareSumKernelUtil {
  static void SquareSum(DeviceCtx* ctx, int64_t n, const T* x, T* y);
  static void MultiSquareSum(DeviceCtx* ctx, const std::vector<SquareSumParam<T>>& params, T* y);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_SQUARE_SUM_KERNEL_UTIL_H_
