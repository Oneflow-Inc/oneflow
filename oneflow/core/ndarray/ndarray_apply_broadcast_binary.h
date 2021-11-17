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
#ifndef ONEFLOW_CORE_NDARRAY_NDARRAY_APPLY_BROADCAST_BINARY_H_
#define ONEFLOW_CORE_NDARRAY_NDARRAY_APPLY_BROADCAST_BINARY_H_

#include "oneflow/core/ndarray/ndarray_apply_broadcast_binary_core.h"
#include "oneflow/core/ndarray/ndarray_apply_binary.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

template<DeviceType device_type, typename T, template<typename> class binary_func,
         typename Enable = void>
struct NdarrayApplyBroadcastBinary;

template<DeviceType device_type, typename T, template<typename> class binary_func>
struct NdarrayApplyBroadcastBinary<
    device_type, T, binary_func,
    typename std::enable_if<std::is_same<T, typename DevDType<device_type, T>::type>::value>::type>
    final {
  using RetT = typename BinaryFuncTrait<binary_func, T>::return_type;
  static void Apply(ep::Stream* stream, const XpuVarNdarray<RetT>& y,
                    const XpuVarNdarray<const T>& a, const XpuVarNdarray<const T>& b) {
    if (a.shape() == b.shape()) {
      return NdarrayApplyBinary<device_type, T, binary_func>::Apply(stream, y, a, b);
    }
    if (TryInplaceApply<std::is_same<RetT, T>::value>(stream, y, a, b)) { return; }
    CheckBroadcastable(y, a, b);
    DimVector simplified_y_dim;
    DimVector simplified_a_dim;
    DimVector simplified_b_dim;
    SimplifyBroadcastShapes(y.shape(), a.shape(), b.shape(), &simplified_y_dim, &simplified_a_dim,
                            &simplified_b_dim);
    return SwitchApply(SwitchCase(simplified_y_dim.size()), stream,
                       XpuVarNdarray<RetT>(Shape(simplified_y_dim), y.ptr()),
                       XpuVarNdarray<const T>(Shape(simplified_a_dim), a.ptr()),
                       XpuVarNdarray<const T>(Shape(simplified_b_dim), b.ptr()));
  }

  template<bool enabled>
  static typename std::enable_if<enabled, bool>::type TryInplaceApply(
      ep::Stream* stream, const XpuVarNdarray<RetT>& y, const XpuVarNdarray<const T>& a,
      const XpuVarNdarray<const T>& b) {
    bool is_inplace = (y.shape() == a.shape() && y.ptr() == a.ptr());
    if (is_inplace) { InplaceApply(stream, y, b); }
    return is_inplace;
  }

  template<bool enabled>
  static typename std::enable_if<!enabled, bool>::type TryInplaceApply(
      ep::Stream* stream, const XpuVarNdarray<RetT>& y, const XpuVarNdarray<const T>& a,
      const XpuVarNdarray<const T>& b) {
    return false;
  }

  static void InplaceApply(ep::Stream* stream, const XpuVarNdarray<T>& y,
                           const XpuVarNdarray<const T>& x) {
    if (y.shape() == x.shape()) {
      return NdarrayApplyBinary<device_type, T, binary_func>::InplaceApply(stream, y, x);
    }
    CheckBroadcastable(y, reinterpret_cast<const XpuVarNdarray<const T>&>(y), x);
    DimVector simplified_y_dim;
    DimVector simplified_x_dim;
    SimplifyBroadcastShapes(y.shape(), x.shape(), &simplified_y_dim, &simplified_x_dim);
    return SwitchInplaceApply(SwitchCase(simplified_y_dim.size()), stream,
                              XpuVarNdarray<T>(Shape(simplified_y_dim), y.ptr()),
                              XpuVarNdarray<const T>(Shape(simplified_x_dim), x.ptr()));
  }

 private:
#define MAKE_NDARRAY_BROADCAST_BINARY(func_name, NDIMS) \
  NdarrayApplyBroadcastBinaryCoreWrapper<device_type, T, NDIMS, binary_func>::func_name
  DEFINE_STATIC_SWITCH_FUNC(void, Apply, MAKE_NDARRAY_BROADCAST_BINARY, MAKE_NDIM_CTRV_SEQ(DIM_SEQ))
#undef MAKE_NDARRAY_BROADCAST_BINARY

#define MAKE_NDARRAY_INPLACE_BROADCAST_BINARY(func_name, NDIMS) \
  NdarrayApplyBroadcastInplaceBinaryCoreWrapper<device_type, T, NDIMS, binary_func>::func_name
  DEFINE_STATIC_SWITCH_FUNC(void, InplaceApply, MAKE_NDARRAY_INPLACE_BROADCAST_BINARY,
                            MAKE_NDIM_CTRV_SEQ(DIM_SEQ))
#undef MAKE_NDARRAY_INPLACE_BROADCAST_BINARY

  static void CheckBroadcastable(
      const XpuVarNdarray<typename BinaryFuncTrait<binary_func, T>::return_type>& y,
      const XpuVarNdarray<const T>& a, const XpuVarNdarray<const T>& b) {
    CHECK_EQ(y.shape().NumAxes(), a.shape().NumAxes());
    CHECK_EQ(y.shape().NumAxes(), b.shape().NumAxes());
    for (int i = 0; i < y.shape().NumAxes(); ++i) {
      CHECK_EQ(y.shape().At(i), (a.shape().At(i) == 0 || b.shape().At(i) == 0)
                                    ? 0
                                    : std::max(a.shape().At(i), b.shape().At(i)));
      if (a.shape().At(i) != b.shape().At(i)) {
        CHECK(a.shape().At(i) == 1 || b.shape().At(i) == 1);
      }
    }
  }
};

template<DeviceType device_type, typename T, template<typename> class binary_func>
struct NdarrayApplyBroadcastBinary<
    device_type, T, binary_func,
    typename std::enable_if<!std::is_same<T, typename DevDType<device_type, T>::type>::value>::type>
    final {
  using NewT = typename DevDType<device_type, T>::type;
  static void Apply(ep::Stream* stream,
                    const XpuVarNdarray<typename BinaryFuncTrait<binary_func, T>::return_type>& y,
                    const XpuVarNdarray<const T>& a, const XpuVarNdarray<const T>& b) {
    return NdarrayApplyBroadcastBinary<device_type, NewT, binary_func>::Apply(
        stream, reinterpret_cast<const XpuVarNdarray<NewT>&>(y),
        reinterpret_cast<const XpuVarNdarray<const NewT>&>(a),
        reinterpret_cast<const XpuVarNdarray<const NewT>&>(b));
  }
  static void InplaceApply(ep::Stream* stream, const XpuVarNdarray<T>& y,
                           const XpuVarNdarray<const T>& x) {
    return NdarrayApplyBroadcastBinary<device_type, NewT, binary_func>::InplaceApply(
        stream, reinterpret_cast<const XpuVarNdarray<NewT>&>(y),
        reinterpret_cast<const XpuVarNdarray<const NewT>&>(x));
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_NDARRAY_APPLY_BROADCAST_BINARY_H_
