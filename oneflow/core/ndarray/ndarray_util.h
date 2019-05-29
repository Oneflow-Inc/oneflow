#ifndef ONEFLOW_CORE_NDARRAY_NDARRAY_UTIL_H_
#define ONEFLOW_CORE_NDARRAY_NDARRAY_UTIL_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/ndarray/xpu_var_ndarray.h"
#include "oneflow/core/ndarray/ndarray_reduce.h"
#include "oneflow/core/ndarray/ndarray_apply_unary.h"
#include "oneflow/core/ndarray/ndarray_apply_broadcast_unary.h"
#include "oneflow/core/ndarray/ndarray_apply_broadcast_binary.h"
#include "oneflow/core/ndarray/xpu_reduced_ndarray.h"
#include "oneflow/core/common/switch_func.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
struct NdarrayUtil final {
  template<template<typename> class unary_func>
  static void BroadcastApply(DeviceCtx* ctx, const XpuVarNdarray<T>& y,
                             const XpuVarNdarray<const T>& x) {
    CHECK_EQ(x.shape().NumAxes(), y.shape().NumAxes());
    return Unary<unary_func>::SwitchBroadcastApply(SwitchCase(x.shape().NumAxes()), ctx, y, x);
  }
  static void BroadcastTo(DeviceCtx* ctx, const XpuVarNdarray<T>& y,
                          const XpuVarNdarray<const T>& x) {
    return BroadcastApply<UnaryFuncIdentity>(ctx, y, x);
  }
  template<template<typename> class binary_func>
  static void BroadcastApply(DeviceCtx* ctx, const XpuVarNdarray<T>& y,
                             const XpuVarNdarray<const T>& a, const XpuVarNdarray<const T>& b) {
    CHECK_EQ(a.shape().NumAxes(), y.shape().NumAxes());
    CHECK_EQ(b.shape().NumAxes(), y.shape().NumAxes());
    return Binary<binary_func>::SwitchBroadcastApply(SwitchCase(y.shape().NumAxes()), ctx, y, a, b);
  }
  static void ReduceSum(DeviceCtx* ctx, const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& x,
                        const XpuVarNdarray<T>& tmp_storage) {
    return NdarrayReduce<device_type, T, BinaryFuncAdd>::Reduce(ctx, y, x, tmp_storage);
  }
  static void ReduceMax(DeviceCtx* ctx, const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& x,
                        const XpuVarNdarray<T>& tmp_storage) {
    return NdarrayReduce<device_type, T, BinaryFuncMax>::Reduce(ctx, y, x, tmp_storage);
  }
  static void ReduceMin(DeviceCtx* ctx, const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& x,
                        const XpuVarNdarray<T>& tmp_storage) {
    return NdarrayReduce<device_type, T, BinaryFuncMin>::Reduce(ctx, y, x, tmp_storage);
  }
  template<template<typename> class unary_func>
  static void ImplaceApplyUnary(DeviceCtx* ctx, const XpuVarNdarray<T>& y) {
    return NdarrayApplyUnary<device_type, T, unary_func>::ImplaceApply(ctx, y);
  }

 private:
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
