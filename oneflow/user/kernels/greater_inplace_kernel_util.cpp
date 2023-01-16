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
#include "oneflow/user/kernels/greater_inplace_kernel_util.h"

namespace oneflow {

template<typename T>
struct GreaterInplaceKernelUtil<DeviceType::kCPU, T> {
  static void Forward(ep::Stream* stream, const int64_t n, const T* x, const T* y, T* out) {
    FOR_RANGE(int64_t, i, 0, n) { out[i] = x[i] > y[i] ? static_cast<T>(1) : static_cast<T>(0); }
  }
};

template<typename T, typename ValueT>
struct ScalarGreaterInplaceKernelUtil<DeviceType::kCPU, T, ValueT> {
  static void Forward(ep::Stream* stream, const int64_t n, const T* x, const Scalar operand,
                      T* out) {
    FOR_RANGE(int64_t, i, 0, n) {
      out[i] =
          x[i] > static_cast<T>(operand.Value<ValueT>()) ? static_cast<T>(1) : static_cast<T>(0);
    }
  }
};

#define INSTANTIATE_GREATER_INPLACE_KERNEL_UTIL_CPU(data_type, other) \
  template struct GreaterInplaceKernelUtil<DeviceType::kCPU, data_type>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_GREATER_INPLACE_KERNEL_UTIL_CPU, GREATER_INPLACE_DATA_TYPE_SEQ_CPU)
#undef INSTANTIATE_GREATER_INPLACE_KERNEL_UTIL_CPU

#define INSTANTIATE_SCALAR_GREATER_INPLACE_KERNEL_UTIL_CPU(data_type, value_data_type)          \
  template struct ScalarGreaterInplaceKernelUtil<DeviceType::kCPU, OF_PP_PAIR_FIRST(data_type), \
                                                 OF_PP_PAIR_FIRST(value_data_type)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_SCALAR_GREATER_INPLACE_KERNEL_UTIL_CPU,
                                 GREATER_INPLACE_DATA_TYPE_SEQ_CPU, SCALAR_VALUE_DATA_TYPE_SEQ)
#undef INSTANTIATE_SCALAR_GREATER_INPLACE_KERNEL_UTIL_CPU

}  // namespace oneflow
