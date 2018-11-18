#ifndef ONEFLOW_CORE_NDARRAY_XPU_NDARRAY_UTIL_H_
#define ONEFLOW_CORE_NDARRAY_XPU_NDARRAY_UTIL_H_

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
struct XpuNdArrayUtil final {
  template<const T (*unary_func)(const T)>
  struct Unary final {
    template<int NDIMS>
    static void BroadcastApply(DeviceCtx* ctx, const XpuVarNdarray<T>& y,
                               const XpuVarNdarray<const T>& x) {
      return NdArrayApplyBroadcastUnary<device_type, T, NDIMS, unary_func>::Apply(ctx, y, x);
    }
#define DEFINE_NDARRAY_BROADCAST_UNARY(func_name, NDIMS) \
  XpuNdArrayUtil<device_type, T>::Unary<unary_func>::func_name<NDIMS>
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
  XpuNdArrayUtil<device_type, T>::Binary<binary_func>::func_name<NDIMS>
    DEFINE_STATIC_SWITCH_FUNC(void, BroadcastApply, DEFINE_NDARRAY_BROADCAST_BINARY,
                              MAKE_NDIM_CTRV_SEQ(DIM_SEQ));
#undef DEFINE_NDARRAY_BROADCAST_BINARY
  };

  template<int NDIMS>
  static void Reduce(DeviceCtx* ctx, const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& x,
                     const XpuVarNdarray<T>& tmp_storage) {
    return NdArrayReduce<device_type, T, NDIMS>::Reduce(ctx, y, x, tmp_storage);
  }
#define DEFINE_NDARRAY_REDUCE(func_name, NDIMS) XpuNdArrayUtil<device_type, T>::func_name<NDIMS>
  DEFINE_STATIC_SWITCH_FUNC(void, Reduce, DEFINE_NDARRAY_REDUCE, MAKE_NDIM_CTRV_SEQ(DIM_SEQ));
#undef DEFINE_NDARRAY_REDUCE

  template<const T (*unary_func)(const T)>
  static void ImplaceApplyUnary(DeviceCtx* ctx, const XpuVarNdarray<T>& y) {
    return NdArrayApplyUnary<device_type, T, unary_func>::ImplaceApply(ctx, y);
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_XPU_NDARRAY_UTIL_H_
