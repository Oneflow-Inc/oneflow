#ifndef ONEFLOW_CORE_NDARRAY_NDARRAY_APPLY_BROADCAST_UNARY_CORE_H_
#define ONEFLOW_CORE_NDARRAY_NDARRAY_APPLY_BROADCAST_UNARY_CORE_H_

#include "oneflow/core/ndarray/ndarray_util.h"
#include "oneflow/core/ndarray/xpu_ndarray_builder.h"
#include "oneflow/core/ndarray/xpu_var_ndarray.h"
#include "oneflow/core/ndarray/unary_func.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T, int NDIMS, const T (*unary_func)(const T)>
struct NdArrayApplyBroadcastUnaryCoreWrapper final {
  static void Apply(DeviceCtx* ctx, const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& x);
};

template<typename T, int NDIMS, const T (*unary_func)(const T)>
struct NdArrayApplyBroadcastUnaryCore final {
  OF_DEVICE_FUNC static void Apply(const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& x) {
    XpuNdArrayBuilder<T, NDIMS> ndarray;
    const auto& x_broadcasted = ndarray.Broadcast(y.shape(), x);
    const auto& ret = ndarray.template Apply<unary_func>(x_broadcasted);
    y.template Assign<NDIMS>(ret);
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_NDARRAY_APPLY_BROADCAST_UNARY_CORE_H_
