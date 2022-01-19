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
#ifndef ONEFLOW_CORE_NDARRAY_NDARRAY_UTIL_H_
#define ONEFLOW_CORE_NDARRAY_NDARRAY_UTIL_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/ndarray/xpu_var_ndarray.h"
#include "oneflow/core/ndarray/xpu_var_ndarray_builder.h"
#include "oneflow/core/ndarray/ndarray_reduce.h"
#include "oneflow/core/ndarray/ndarray_apply_unary.h"
#include "oneflow/core/ndarray/ndarray_apply_binary.h"
#include "oneflow/core/ndarray/ndarray_apply_broadcast_unary.h"
#include "oneflow/core/ndarray/ndarray_apply_broadcast_binary.h"
#include "oneflow/core/ndarray/xpu_reduced_ndarray.h"
#include "oneflow/core/ndarray/xpu_ndarray_assign.h"
#include "oneflow/core/common/switch_func.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
struct NdarrayUtil final {
  static XpuVarNdarrayBuilder<const T> GetValNdarrayBuilder() {
    return XpuVarNdarrayBuilder<const T>();
  }
  static XpuVarNdarrayBuilder<T> GetVarNdarrayBuilder() { return XpuVarNdarrayBuilder<T>(); }

  static void Assign(ep::Stream* stream, const XpuVarNdarray<T>& y,
                     const XpuVarNdarray<const T>& x) {
    return XpuNdarrayAssign<device_type, T>::Assign(stream, y, x);
  }

  static void BroadcastTo(ep::Stream* stream, const XpuVarNdarray<T>& y,
                          const XpuVarNdarray<const T>& x) {
    return BroadcastIdentity(stream, y, x);
  }

#define DEFINE_UNARY_FUNC(func_name)                                                         \
  static void func_name(                                                                     \
      ep::Stream* stream,                                                                    \
      const XpuVarNdarray<typename UnaryFuncTrait<UnaryFunc##func_name, T>::return_type>& y, \
      const XpuVarNdarray<const T>& x) {                                                     \
    return ApplyUnary<UnaryFunc##func_name>(stream, y, x);                                   \
  }
  OF_PP_FOR_EACH_ATOMIC(DEFINE_UNARY_FUNC, ARITHMETIC_UNARY_FUNC_NAME_SEQ)
#undef DEFINE_UNARY_FUNC

#define DEFINE_ARITHMETIC_BINARY_FUNC(func_name)                                               \
  static void func_name(                                                                       \
      ep::Stream* stream,                                                                      \
      const XpuVarNdarray<typename BinaryFuncTrait<BinaryFunc##func_name, T>::return_type>& y, \
      const XpuVarNdarray<const T>& a, const XpuVarNdarray<const T>& b) {                      \
    return ApplyBinary<BinaryFunc##func_name>(stream, y, a, b);                                \
  }
  OF_PP_FOR_EACH_ATOMIC(DEFINE_ARITHMETIC_BINARY_FUNC, ARITHMETIC_BINARY_FUNC_NAME_SEQ)
#undef DEFINE_ARITHMETIC_BINARY_FUNC

#define DEFINE_LOGICAL_BINARY_FUNC(func_name)                                                  \
  static void func_name(                                                                       \
      ep::Stream* stream,                                                                      \
      const XpuVarNdarray<typename BinaryFuncTrait<BinaryFunc##func_name, T>::return_type>& y, \
      const XpuVarNdarray<const T>& a, const XpuVarNdarray<const T>& b) {                      \
    return ApplyBinary<BinaryFunc##func_name>(stream, y, a, b);                                \
  }
  OF_PP_FOR_EACH_ATOMIC(DEFINE_LOGICAL_BINARY_FUNC, LOGICAL_BINARY_FUNC_NAME_SEQ)
#undef DEFINE_LOGICAL_BINARY_FUNC

#define DEFINE_BROADCAST_UNARY_FUNC(func_name)                                               \
  static void Broadcast##func_name(                                                          \
      ep::Stream* stream,                                                                    \
      const XpuVarNdarray<typename UnaryFuncTrait<UnaryFunc##func_name, T>::return_type>& y, \
      const XpuVarNdarray<const T>& x) {                                                     \
    return BroadcastApplyUnary<UnaryFunc##func_name>(stream, y, x);                          \
  }
  OF_PP_FOR_EACH_ATOMIC(DEFINE_BROADCAST_UNARY_FUNC, ARITHMETIC_UNARY_FUNC_NAME_SEQ)
#undef DEFINE_BROADCAST_UNARY_FUNC

#define DEFINE_BROADCAST_ARITHMETIC_BINARY_FUNC(func_name)                                     \
  static void Broadcast##func_name(                                                            \
      ep::Stream* stream,                                                                      \
      const XpuVarNdarray<typename BinaryFuncTrait<BinaryFunc##func_name, T>::return_type>& y, \
      const XpuVarNdarray<const T>& a, const XpuVarNdarray<const T>& b) {                      \
    return BroadcastApplyBinary<BinaryFunc##func_name>(stream, y, a, b);                       \
  }
  OF_PP_FOR_EACH_ATOMIC(DEFINE_BROADCAST_ARITHMETIC_BINARY_FUNC, ARITHMETIC_BINARY_FUNC_NAME_SEQ)
#undef DEFINE_BROADCAST_ARITHMETIC_BINARY_FUNC

#define DEFINE_BROADCAST_LOGICAL_BINARY_FUNC(func_name)                                        \
  static void Broadcast##func_name(                                                            \
      ep::Stream* stream,                                                                      \
      const XpuVarNdarray<typename BinaryFuncTrait<BinaryFunc##func_name, T>::return_type>& y, \
      const XpuVarNdarray<const T>& a, const XpuVarNdarray<const T>& b) {                      \
    return BroadcastApplyBinary<BinaryFunc##func_name>(stream, y, a, b);                       \
  }
  OF_PP_FOR_EACH_ATOMIC(DEFINE_BROADCAST_LOGICAL_BINARY_FUNC, LOGICAL_BINARY_FUNC_NAME_SEQ)
#undef DEFINE_BROADCAST_LOGICAL_BINARY_FUNC

#define DEFINE_INPLACE_UNARY_FUNC(func_name)                                      \
  static void Inplace##func_name(ep::Stream* stream, const XpuVarNdarray<T>& y) { \
    InplaceApply<UnaryFunc##func_name>(stream, y);                                \
  }
  OF_PP_FOR_EACH_ATOMIC(DEFINE_INPLACE_UNARY_FUNC, ARITHMETIC_UNARY_FUNC_NAME_SEQ)
#undef DEFINE_INPLACE_UNARY_FUNC

#define DEFINE_INPLACE_BINARY_FUNC(func_name)                                   \
  static void Inplace##func_name(ep::Stream* stream, const XpuVarNdarray<T>& y, \
                                 const XpuVarNdarray<const T>& x) {             \
    InplaceApply<BinaryFunc##func_name>(stream, y, x);                          \
  }
  OF_PP_FOR_EACH_ATOMIC(DEFINE_INPLACE_BINARY_FUNC, ARITHMETIC_BINARY_FUNC_NAME_SEQ)
#undef DEFINE_INPLACE_BINARY_FUNC

#define DEFINE_INPLACE_BROADCAST_BINARY_FUNC(func_name)                                  \
  static void InplaceBroadcast##func_name(ep::Stream* stream, const XpuVarNdarray<T>& y, \
                                          const XpuVarNdarray<const T>& x) {             \
    return InplaceBroadcastApply<BinaryFunc##func_name>(stream, y, x);                   \
  }
  OF_PP_FOR_EACH_ATOMIC(DEFINE_INPLACE_BROADCAST_BINARY_FUNC, ARITHMETIC_BINARY_FUNC_NAME_SEQ)
#undef DEFINE_INPLACE_BROADCAST_BINARY_FUNC

#define DEFINE_REDUCE_FUNC(func_name)                                                 \
  static void Reduce##func_name(ep::Stream* stream, const XpuVarNdarray<T>& y,        \
                                const XpuVarNdarray<const T>& x,                      \
                                const XpuVarNdarray<T>& tmp_storage) {                \
    return NdarrayReduce<device_type, T, BinaryFunc##func_name>::Reduce(stream, y, x, \
                                                                        tmp_storage); \
  }
  OF_PP_FOR_EACH_ATOMIC(DEFINE_REDUCE_FUNC, REDUCE_BINARY_FUNC_NAME_SEQ)
#undef DEFINE_REDUCE_FUNC

 private:
  template<template<typename> class unary_func>
  static void BroadcastApplyUnary(
      ep::Stream* stream,
      const XpuVarNdarray<typename UnaryFuncTrait<unary_func, T>::return_type>& y,
      const XpuVarNdarray<const T>& x) {
    CHECK_EQ(x.shape().NumAxes(), y.shape().NumAxes());
    return NdarrayApplyBroadcastUnary<device_type, T, unary_func>::Apply(stream, y, x);
  }

  template<template<typename> class binary_func>
  static void BroadcastApplyBinary(
      ep::Stream* stream,
      const XpuVarNdarray<typename BinaryFuncTrait<binary_func, T>::return_type>& y,
      const XpuVarNdarray<const T>& a, const XpuVarNdarray<const T>& b) {
    CHECK_EQ(a.shape().NumAxes(), y.shape().NumAxes());
    CHECK_EQ(b.shape().NumAxes(), y.shape().NumAxes());
    return NdarrayApplyBroadcastBinary<device_type, T, binary_func>::Apply(stream, y, a, b);
  }

  template<template<typename> class binary_func>
  static void InplaceBroadcastApply(ep::Stream* stream, const XpuVarNdarray<T>& y,
                                    const XpuVarNdarray<const T>& x) {
    static_assert(std::is_same<T, typename BinaryFuncTrait<binary_func, T>::return_type>::value,
                  "T must be same with BinaryFuncTrait<binary_func, T>::return_type");
    CHECK_EQ(x.shape().NumAxes(), y.shape().NumAxes());
    return NdarrayApplyBroadcastBinary<device_type, T, binary_func>::InplaceApply(stream, y, x);
  }

  template<template<typename> class unary_func>
  static void InplaceApply(ep::Stream* stream, const XpuVarNdarray<T>& y) {
    static_assert(std::is_same<T, typename UnaryFuncTrait<unary_func, T>::return_type>::value,
                  "T must be same with UnaryFuncTrait<unary_func, T>::return_type");
    return NdarrayApplyUnary<device_type, T, unary_func>::InplaceApply(stream, y);
  }

  template<template<typename> class binary_func>
  static void InplaceApply(ep::Stream* stream, const XpuVarNdarray<T>& y,
                           const XpuVarNdarray<const T>& x) {
    static_assert(std::is_same<T, typename BinaryFuncTrait<binary_func, T>::return_type>::value,
                  "T must be same with BinaryFuncTrait<binary_func, T>::return_type");
    return NdarrayApplyBinary<device_type, T, binary_func>::InplaceApply(stream, y, x);
  }

  template<template<typename> class unary_func>
  static void ApplyUnary(
      ep::Stream* stream,
      const XpuVarNdarray<typename UnaryFuncTrait<unary_func, T>::return_type>& y,
      const XpuVarNdarray<const T>& x) {
    return NdarrayApplyUnary<device_type, T, unary_func>::Apply(stream, y, x);
  }

  template<template<typename> class binary_func>
  static void ApplyBinary(
      ep::Stream* stream,
      const XpuVarNdarray<typename BinaryFuncTrait<binary_func, T>::return_type>& y,
      const XpuVarNdarray<const T>& a, const XpuVarNdarray<const T>& b) {
    if (a.host_ptr() == y.host_ptr()) {
      CHECK(a.host_shape() == y.host_shape());
      return NdarrayApplyBinary<device_type, T, binary_func>::InplaceApply(stream, y, b);
    } else {
      return NdarrayApplyBinary<device_type, T, binary_func>::Apply(stream, y, a, b);
    }
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_NDARRAY_UTIL_H_
