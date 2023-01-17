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
#ifndef ONEFLOW_USER_KERNELS_GREATER_INPLACE_KERNEL_UTIL_H_
#define ONEFLOW_USER_KERNELS_GREATER_INPLACE_KERNEL_UTIL_H_

#include "oneflow/core/common/scalar.h"
#include "oneflow/core/common/shape_view.h"
#include "oneflow/core/ep/include/stream.h"
#include "oneflow/core/ndarray/ndarray_util.h"

namespace oneflow {

template<DeviceType device_type, typename T, typename ValueT>
struct ScalarGreaterInplaceKernelUtil {
  static void Forward(ep::Stream* stream, const int64_t n, const T* x, const Scalar operand,
                      T* out);
};

template<DeviceType device_type, typename T>
struct GreaterInplaceKernelUtil {
  static void Forward(ep::Stream* stream, const int64_t n, const T* x, const T* y, T* out);
  static void YBroadcastToX(ep::Stream* stream, const int64_t n, const T* x, const T* y,
                            T* broadcast_y, const ShapeView& x_shape, const ShapeView& y_shape) {
    const int64_t x_ndim = x_shape.NumAxes();
    const int64_t y_ndim = y_shape.NumAxes();
    const int64_t num_prepend = x_ndim - y_ndim;
    std::vector<int64_t> prepend_shape(num_prepend, 1);
    std::vector<int32_t> broadcast_axes;
    for (int i = 0; i < y_ndim; ++i) { prepend_shape.emplace_back(y_shape.At(i)); }
    for (int i = 0; i < num_prepend; ++i) { broadcast_axes.emplace_back(i); }
    for (int i = num_prepend; i < prepend_shape.size(); ++i) {
      if (prepend_shape[i] != x_shape.At(i)) {
        if (prepend_shape[i] == 1) { broadcast_axes.emplace_back(i); }
      }
    }
    const Shape& reduced_shape =
        CreateReducedShapeOrOnesShape(x_shape, {broadcast_axes.begin(), broadcast_axes.end()});
    NdarrayUtil<device_type, T>::BroadcastTo(stream, XpuVarNdarray<T>(x_shape, broadcast_y),
                                             XpuVarNdarray<const T>(reduced_shape, y));
  }
};

#define SCALAR_VALUE_DATA_TYPE_SEQ                \
  OF_PP_MAKE_TUPLE_SEQ(int64_t, DataType::kInt64) \
  OF_PP_MAKE_TUPLE_SEQ(double, DataType::kDouble)

#define GREATER_INPLACE_DATA_TYPE_SEQ_CPU \
  FLOATING_DATA_TYPE_SEQ                  \
  SIGNED_INT_DATA_TYPE_SEQ

#ifdef WITH_CUDA
#define GREATER_INPLACE_DATA_TYPE_SEQ_CUDA \
  FLOATING_DATA_TYPE_SEQ                   \
  SIGNED_INT_DATA_TYPE_SEQ                 \
  HALF_DATA_TYPE_SEQ
#endif

}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_GREATER_INPLACE_KERNEL_UTIL_H_
