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
#ifndef ONEFLOW_CORE_NDARRAY_XPU_ASSIGN_H_
#define ONEFLOW_CORE_NDARRAY_XPU_ASSIGN_H_

#include "oneflow/core/ndarray/ndarray_assign_core.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/ep/include/primitive/elementwise_unary.h"
#include "oneflow/core/ep/include/primitive/unary_op.h"

namespace oneflow {

template<DeviceType device_type, typename T, typename Enable = void>
struct XpuNdarrayAssign;

template<DeviceType device_type, typename T>
struct XpuNdarrayAssign<
    device_type, T,
    typename std::enable_if<std::is_same<T, typename DevDType<device_type, T>::type>::value>::type>
    final {
  template<int NDIMS, typename X>
  static void Assign(ep::Stream* stream, const XpuVarNdarray<T>& y,
                     const XpuReducedNdarray<X, NDIMS>& reduced) {
    NdarrayAssignCoreWrapper<device_type, T, X, NDIMS>::Assign(stream, y, reduced);
  }
  template<int NDIMS, typename X>
  static void Assign(ep::Stream* stream, const XpuVarNdarray<T>& y,
                     const XpuVarNdarray<const X>& x) {
    NdarrayAssignCoreWrapper<device_type, T, X, NDIMS>::Assign(stream, y, x);
  }
  static void Assign(ep::Stream* stream, const XpuVarNdarray<T>& y,
                     const XpuVarNdarray<const T>& x) {
    CHECK(y.shape() == x.shape());
    if (x.ptr() == y.ptr()) { return; }
    Memcpy<device_type>(stream, y.ptr(), x.ptr(), y.shape().ElemNum() * sizeof(T));
  }

  static void AssignNanSum(ep::Stream* stream, const XpuVarNdarray<T>& y,
                           const XpuVarNdarray<const T>& x) {
    CHECK(y.shape() == x.shape());  // NOLINT
    CHECK_EQ(device_type, stream->device_type()) << "Device type mismatch";
    std::unique_ptr<ep::primitive::ElementwiseUnary> primitive =
        ep::primitive::NewPrimitive<ep::primitive::ElementwiseUnaryFactory>(
            device_type, ep::primitive::UnaryOp::kNanAssign, GetDataType<T>(), GetDataType<T>());
    CHECK(primitive) << "Can not create NanSum primitive for device type " << device_type;
    primitive->Launch(stream, x.ptr(), y.ptr(), y.shape().ElemNum());
  }
};

template<DeviceType device_type, typename T>
struct XpuNdarrayAssign<
    device_type, T,
    typename std::enable_if<!std::is_same<T, typename DevDType<device_type, T>::type>::value>::type>
    final {
  using NewT = typename DevDType<device_type, T>::type;
  template<int NDIMS>
  static void Assign(ep::Stream* stream, const XpuVarNdarray<T>& y,
                     const XpuReducedNdarray<T, NDIMS>& reduced) {
    XpuNdarrayAssign<device_type, NewT>::Assign(
        stream, reinterpret_cast<const XpuVarNdarray<NewT>&>(y),
        reinterpret_cast<const XpuReducedNdarray<NewT, NDIMS>&>(reduced));
  }

  static void Assign(ep::Stream* ctx, const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& x) {
    XpuNdarrayAssign<device_type, NewT>::Assign(
        ctx, reinterpret_cast<const XpuVarNdarray<NewT>&>(y),
        reinterpret_cast<const XpuVarNdarray<const NewT>&>(x));
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_XPU_ASSIGN_H_
