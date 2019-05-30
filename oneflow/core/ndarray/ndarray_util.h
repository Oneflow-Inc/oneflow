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

  static void Assign(DeviceCtx* ctx, const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& x) {
    return XpuNdarrayAssign<device_type, T>::Assign(ctx, y, x);
  }

  static void BroadcastTo(DeviceCtx* ctx, const XpuVarNdarray<T>& y,
                          const XpuVarNdarray<const T>& x) {
    return BroadcastIdentity(ctx, y, x);
  }

#define INSTANTIATE_UNARY_FUNC(func_name)                          \
  static void func_name(DeviceCtx* ctx, const XpuVarNdarray<T>& y, \
                        const XpuVarNdarray<const T>& x) {         \
    return Apply<UnaryFunc##func_name>(ctx, y, x);                 \
  }
  OF_PP_FOR_EACH_ATOMIC(INSTANTIATE_UNARY_FUNC, ARITHMETIC_UNARY_FUNC_NAME_SEQ)
#undef INSTANTIATE_UNARY_FUNC

#define INSTANTIATE_BINARY_FUNC(func_name)                                                  \
  static void func_name(DeviceCtx* ctx, const XpuVarNdarray<T>& y,                          \
                        const XpuVarNdarray<const T>& a, const XpuVarNdarray<const T>& b) { \
    return Apply<BinaryFunc##func_name>(ctx, y, a, b);                                      \
  }
  OF_PP_FOR_EACH_ATOMIC(INSTANTIATE_BINARY_FUNC, ARITHMETIC_BINARY_FUNC_NAME_SEQ)
#undef INSTANTIATE_BINARY_FUNC

#define INSTANTIATE_BROADCAST_UNARY_FUNC(func_name)                           \
  static void Broadcast##func_name(DeviceCtx* ctx, const XpuVarNdarray<T>& y, \
                                   const XpuVarNdarray<const T>& x) {         \
    return BroadcastApply<UnaryFunc##func_name>(ctx, y, x);                   \
  }
  OF_PP_FOR_EACH_ATOMIC(INSTANTIATE_BROADCAST_UNARY_FUNC, ARITHMETIC_UNARY_FUNC_NAME_SEQ)
#undef INSTANTIATE_BROADCAST_UNARY_FUNC

#define INSTANTIATE_BROADCAST_BINARY_FUNC(func_name)                          \
  static void Broadcast##func_name(DeviceCtx* ctx, const XpuVarNdarray<T>& y, \
                                   const XpuVarNdarray<const T>& a,           \
                                   const XpuVarNdarray<const T>& b) {         \
    return BroadcastApply<BinaryFunc##func_name>(ctx, y, a, b);               \
  }
  OF_PP_FOR_EACH_ATOMIC(INSTANTIATE_BROADCAST_BINARY_FUNC, ARITHMETIC_BINARY_FUNC_NAME_SEQ)
#undef INSTANTIATE_BROADCAST_BINARY_FUNC

#define INSTANTIATE_INPLACE_UNARY_FUNC(func_name)                             \
  static void Inplace##func_name(DeviceCtx* ctx, const XpuVarNdarray<T>& y) { \
    InplaceApply<UnaryFunc##func_name>(ctx, y);                               \
  }
  OF_PP_FOR_EACH_ATOMIC(INSTANTIATE_INPLACE_UNARY_FUNC, ARITHMETIC_UNARY_FUNC_NAME_SEQ)
#undef INSTANTIATE_INPLACE_UNARY_FUNC

#define INSTANTIATE_INPLACE_BINARY_FUNC(func_name)                          \
  static void Inplace##func_name(DeviceCtx* ctx, const XpuVarNdarray<T>& y, \
                                 const XpuVarNdarray<const T>& x) {         \
    InplaceApply<BinaryFunc##func_name>(ctx, y, x);                         \
  }
  OF_PP_FOR_EACH_ATOMIC(INSTANTIATE_INPLACE_BINARY_FUNC, ARITHMETIC_BINARY_FUNC_NAME_SEQ)
#undef INSTANTIATE_INPLACE_BINARY_FUNC

#define INSTANTIATE_REDUCE_FUNC(func_name)                                                       \
  static void Reduce##func_name(DeviceCtx* ctx, const XpuVarNdarray<T>& y,                       \
                                const XpuVarNdarray<const T>& x,                                 \
                                const XpuVarNdarray<T>& tmp_storage) {                           \
    return NdarrayReduce<device_type, T, BinaryFunc##func_name>::Reduce(ctx, y, x, tmp_storage); \
  }
  OF_PP_FOR_EACH_ATOMIC(INSTANTIATE_REDUCE_FUNC, REDUCE_BINARY_FUNC_NAME_SEQ)
#undef INSTANTIATE_REDUCE_FUNC

 private:
  template<template<typename> class unary_func>
  static void BroadcastApply(DeviceCtx* ctx, const XpuVarNdarray<T>& y,
                             const XpuVarNdarray<const T>& x) {
    CHECK_EQ(x.shape().NumAxes(), y.shape().NumAxes());
    return Unary<unary_func>::SwitchBroadcastApply(SwitchCase(x.shape().NumAxes()), ctx, y, x);
  }

  template<template<typename> class binary_func>
  static void BroadcastApply(DeviceCtx* ctx, const XpuVarNdarray<T>& y,
                             const XpuVarNdarray<const T>& a, const XpuVarNdarray<const T>& b) {
    CHECK_EQ(a.shape().NumAxes(), y.shape().NumAxes());
    CHECK_EQ(b.shape().NumAxes(), y.shape().NumAxes());
    return Binary<binary_func>::SwitchBroadcastApply(SwitchCase(y.shape().NumAxes()), ctx, y, a, b);
  }

  template<template<typename> class unary_func>
  static void InplaceApply(DeviceCtx* ctx, const XpuVarNdarray<T>& y) {
    return NdarrayApplyUnary<device_type, T, unary_func>::InplaceApply(ctx, y);
  }

  template<template<typename> class binary_func>
  static void InplaceApply(DeviceCtx* ctx, const XpuVarNdarray<T>& y,
                           const XpuVarNdarray<const T>& x) {
    return NdarrayApplyBinary<device_type, T, binary_func>::InplaceApply(ctx, y, x);
  }

  template<template<typename> class unary_func>
  static void Apply(DeviceCtx* ctx, const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& x) {
    return NdarrayApplyUnary<device_type, T, unary_func>::Apply(ctx, y, x);
  }

  template<template<typename> class binary_func>
  static void Apply(DeviceCtx* ctx, const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& a,
                    const XpuVarNdarray<const T>& b) {
    return NdarrayApplyBinary<device_type, T, binary_func>::Apply(ctx, y, a, b);
  }

  template<template<typename> class unary_func>
  struct Unary final {
    template<int NDIMS>
    static void BroadcastApply(DeviceCtx* ctx, const XpuVarNdarray<T>& y,
                               const XpuVarNdarray<const T>& x) {
      return NdarrayApplyBroadcastUnary<device_type, T, NDIMS, unary_func>::Apply(ctx, y, x);
    }
#define DEFINE_NDARRAY_BROADCAST_UNARY(func_name, NDIMS) \
  NdarrayUtil<device_type, T>::Unary<unary_func>::func_name<NDIMS>
    DEFINE_STATIC_SWITCH_FUNC(void, BroadcastApply, DEFINE_NDARRAY_BROADCAST_UNARY,
                              MAKE_NDIM_CTRV_SEQ(DIM_SEQ));
#undef DEFINE_NDARRAY_BROADCAST_UNARY
  };

  template<template<typename> class binary_func>
  struct Binary final {
    template<int NDIMS>
    static void BroadcastApply(DeviceCtx* ctx, const XpuVarNdarray<T>& y,
                               const XpuVarNdarray<const T>& a, const XpuVarNdarray<const T>& b) {
      return NdarrayApplyBroadcastBinary<device_type, T, NDIMS, binary_func>::Apply(ctx, y, a, b);
    }
#define DEFINE_NDARRAY_BROADCAST_BINARY(func_name, NDIMS) \
  NdarrayUtil<device_type, T>::Binary<binary_func>::func_name<NDIMS>
    DEFINE_STATIC_SWITCH_FUNC(void, BroadcastApply, DEFINE_NDARRAY_BROADCAST_BINARY,
                              MAKE_NDIM_CTRV_SEQ(DIM_SEQ));
#undef DEFINE_NDARRAY_BROADCAST_BINARY
  };
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_NDARRAY_UTIL_H_
