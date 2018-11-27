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
  template<const T (*unary_func)(const T)>
  static void BroadcastApply(DeviceCtx* ctx, const XpuVarNdarray<T>& y,
                             const XpuVarNdarray<const T>& x) {
    CHECK_EQ(x.shape().NumAxes(), y.shape().NumAxes());
    return Unary<unary_func>::SwitchBroadcastApply(SwitchCase(x.shape().NumAxes()), ctx, y, x);
  }
  template<const T (*binary_func)(const T, const T)>
  static void BroadcastApply(DeviceCtx* ctx, const XpuVarNdarray<T>& y,
                             const XpuVarNdarray<const T>& a, const XpuVarNdarray<const T>& b) {
    CHECK_EQ(a.shape().NumAxes(), y.shape().NumAxes());
    CHECK_EQ(b.shape().NumAxes(), y.shape().NumAxes());
    return Binary<binary_func>::SwitchBroadcastApply(SwitchCase(y.shape().NumAxes()), ctx, y, a, b);
  }

  static void ReduceSum(DeviceCtx* ctx, const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& x,
                        const XpuVarNdarray<T>& tmp_storage) {
    return NdArrayReduce<device_type, T, BinaryFuncAdd>::Reduce(ctx, y, x, tmp_storage);
  }

  template<const T (*unary_func)(const T)>
  static void ImplaceApplyUnary(DeviceCtx* ctx, const XpuVarNdarray<T>& y) {
    return NdArrayApplyUnary<device_type, T, unary_func>::ImplaceApply(ctx, y);
  }

 private:
  template<const T (*unary_func)(const T)>
  struct Unary final {
    template<int NDIMS>
    static void BroadcastApply(DeviceCtx* ctx, const XpuVarNdarray<T>& y,
                               const XpuVarNdarray<const T>& x) {
      return NdArrayApplyBroadcastUnary<device_type, T, NDIMS, unary_func>::Apply(ctx, y, x);
    }
#define DEFINE_NDARRAY_BROADCAST_UNARY(func_name, NDIMS) \
  NdarrayUtil<device_type, T>::Unary<unary_func>::func_name<NDIMS>
    DEFINE_STATIC_SWITCH_FUNC(void, BroadcastApply, DEFINE_NDARRAY_BROADCAST_UNARY,
                              MAKE_NDIM_CTRV_SEQ(DIM_SEQ));
#undef DEFINE_NDARRAY_BROADCAST_UNARY
  };

  template<const T (*binary_func)(const T, const T)>
  struct Binary final {
    template<int NDIMS>
    static void BroadcastApply(DeviceCtx* ctx, const XpuVarNdarray<T>& y,
                               const XpuVarNdarray<const T>& a, const XpuVarNdarray<const T>& b) {
      return NdArrayApplyBroadcastBinary<device_type, T, NDIMS, binary_func>::Apply(ctx, y, a, b);
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
