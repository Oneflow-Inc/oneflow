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
#ifndef ONEFLOW_CORE_NDARRAY_NDARRAY_APPLY_BROADCAST_UNARY_H_
#define ONEFLOW_CORE_NDARRAY_NDARRAY_APPLY_BROADCAST_UNARY_H_

#include "oneflow/core/ndarray/ndarray_apply_broadcast_unary_core.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

template<DeviceType device_type, typename T, template<typename> class unary_func,
         typename Enable = void>
struct NdarrayApplyBroadcastUnary;

template<DeviceType device_type, typename T, template<typename> class unary_func>
struct NdarrayApplyBroadcastUnary<
    device_type, T, unary_func,
    typename std::enable_if<std::is_same<T, typename DevDType<device_type, T>::type>::value>::type>
    final {
  static void Apply(ep::Stream* stream, const XpuVarNdarray<T>& y,
                    const XpuVarNdarray<const T>& x) {
    CheckBroadcastable(y, x);
    DimVector simplified_y_dim;
    DimVector simplified_x_dim;
    SimplifyBroadcastShapes(y.shape(), x.shape(), &simplified_y_dim, &simplified_x_dim);
    SwitchApply(SwitchCase(simplified_y_dim.size()), stream,
                XpuVarNdarray<T>(Shape(simplified_y_dim), y.ptr()),
                XpuVarNdarray<const T>(Shape(simplified_x_dim), x.ptr()));
  }

 private:
#define DEFINE_NDARRAY_BROADCAST_UNARY(func_name, NDIMS) \
  NdarrayApplyBroadcastUnaryCoreWrapper<device_type, T, NDIMS, unary_func>::func_name
  DEFINE_STATIC_SWITCH_FUNC(void, Apply, DEFINE_NDARRAY_BROADCAST_UNARY,
                            MAKE_NDIM_CTRV_SEQ(DIM_SEQ))
#undef DEFINE_NDARRAY_BROADCAST_UNARY
  static void CheckBroadcastable(const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& x) {
    CHECK_EQ(y.shape().NumAxes(), x.shape().NumAxes());
    for (int i = 0; i < y.shape().NumAxes(); ++i) {
      CHECK(x.shape().At(i) == 1 || x.shape().At(i) == y.shape().At(i));
    }
  }
};

template<DeviceType device_type, typename T, template<typename> class unary_func>
struct NdarrayApplyBroadcastUnary<
    device_type, T, unary_func,
    typename std::enable_if<!std::is_same<T, typename DevDType<device_type, T>::type>::value>::type>
    final {
  static void Apply(ep::Stream* stream, const XpuVarNdarray<T>& y,
                    const XpuVarNdarray<const T>& x) {
    using NewT = typename DevDType<device_type, T>::type;
    return NdarrayApplyBroadcastUnary<device_type, NewT, unary_func>::Apply(
        stream, reinterpret_cast<const XpuVarNdarray<NewT>&>(y),
        reinterpret_cast<const XpuVarNdarray<const NewT>&>(x));
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_NDARRAY_APPLY_BROADCAST_UNARY_H_
