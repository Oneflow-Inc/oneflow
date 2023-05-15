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
#include "oneflow/user/kernels/lerp_kernel_util.h"

namespace oneflow {

template<typename T>
struct LerpKernelUtil<DeviceType::kCPU, T> {
  static void Forward(ep::Stream* stream, const int64_t n, const T* start, const T* weight,
                      const T* end, T* out) {
    FOR_RANGE(int64_t, i, 0, n) { out[i] = start[i] + weight[i] * (end[i] - start[i]); }
  }

  static void Backward(ep::Stream* stream, const int64_t n, const T* start, const T* weight,
                       const T* end, const T* out_diff, T* start_diff, T* weight_diff,
                       T* end_diff) {
    FOR_RANGE(int64_t, i, 0, n) {
      T out_diff_i = out_diff[i];
      start_diff[i] = (static_cast<T>(1.0) - weight[i]) * out_diff_i;
      weight_diff[i] = (end[i] - start[i]) * out_diff_i;
      end_diff[i] = weight[i] * out_diff_i;
    }
  }
};

template<typename T, typename ValueT>
struct ScalarLerpKernelUtil<DeviceType::kCPU, T, ValueT> {
  static void Forward(ep::Stream* stream, const int64_t n, const T* start, const T* end,
                      const Scalar operand, T* out) {
    T weight = static_cast<T>(operand.Value<ValueT>());
    FOR_RANGE(int64_t, i, 0, n) { out[i] = start[i] + weight * (end[i] - start[i]); }
  }

  static void Backward(ep::Stream* stream, const int64_t n, const T* start, const T* end,
                       const T* out_diff, const Scalar operand, T* start_diff, T* end_diff) {
    T weight = static_cast<T>(operand.Value<ValueT>());
    FOR_RANGE(int64_t, i, 0, n) {
      T out_diff_i = out_diff[i];
      start_diff[i] = (static_cast<T>(1.0) - weight) * out_diff_i;
      end_diff[i] = out_diff_i - start_diff[i];
    }
  }
};

#define INSTANTIATE_LERP_KERNEL_UTIL_CPU(data_type, other) \
  template struct LerpKernelUtil<DeviceType::kCPU, data_type>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_LERP_KERNEL_UTIL_CPU, LERP_DATA_TYPE_SEQ_CPU)
#undef INSTANTIATE_LERP_KERNEL_UTIL_CPU

#define INSTANTIATE_SCALAR_LERP_KERNEL_UTIL_CPU(data_type, value_data_type)           \
  template struct ScalarLerpKernelUtil<DeviceType::kCPU, OF_PP_PAIR_FIRST(data_type), \
                                       OF_PP_PAIR_FIRST(value_data_type)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_SCALAR_LERP_KERNEL_UTIL_CPU, LERP_DATA_TYPE_SEQ_CPU,
                                 SCALAR_VALUE_DATA_TYPE_SEQ)
#undef INSTANTIATE_SCALAR_LERP_KERNEL_UTIL_CPU

}  // namespace oneflow
