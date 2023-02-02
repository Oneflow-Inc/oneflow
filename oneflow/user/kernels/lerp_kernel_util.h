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
#ifndef ONEFLOW_USER_KERNELS_LERP_KERNEL_UTIL_H_
#define ONEFLOW_USER_KERNELS_LERP_KERNEL_UTIL_H_

#include "oneflow/core/common/scalar.h"
#include "oneflow/core/common/shape_view.h"
#include "oneflow/core/ep/include/stream.h"
#include "oneflow/core/ndarray/ndarray_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
struct LerpKernelUtil {
  static void Forward(ep::Stream* stream, const int64_t n, const T* start, const T* weight,
                      const T* end, T* out);
  static void Backward(ep::Stream* stream, const int64_t n, const T* start, const T* weight,
                       const T* end, const T* out_diff, T* start_diff, T* weight_diff, T* end_diff);
};

template<DeviceType device_type, typename T, typename ValueT>
struct ScalarLerpKernelUtil {
  static void Forward(ep::Stream* stream, const int64_t n, const T* start, const T* end,
                      const Scalar operand, T* out);
  static void Backward(ep::Stream* stream, const int64_t n, const T* start, const T* end,
                       const T* out_diff, const Scalar operand, T* start_diff, T* end_diff);
};

#define SCALAR_VALUE_DATA_TYPE_SEQ                \
  OF_PP_MAKE_TUPLE_SEQ(int64_t, DataType::kInt64) \
  OF_PP_MAKE_TUPLE_SEQ(double, DataType::kDouble)

#define LERP_DATA_TYPE_SEQ_CPU \
  FLOATING_DATA_TYPE_SEQ       \
  SIGNED_INT_DATA_TYPE_SEQ     \
  UNSIGNED_INT_DATA_TYPE_SEQ

#ifdef WITH_CUDA
#define LERP_DATA_TYPE_SEQ_CUDA \
  FLOATING_DATA_TYPE_SEQ        \
  SIGNED_INT_DATA_TYPE_SEQ      \
  UNSIGNED_INT_DATA_TYPE_SEQ    \
  HALF_DATA_TYPE_SEQ
#endif

}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_LERP_KERNEL_UTIL_H_
