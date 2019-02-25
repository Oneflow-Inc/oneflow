#ifndef ONEFLOW_CORE_NDARRAY_NDARRAY_APPLY_BROADCAST_UNARY_H_
#define ONEFLOW_CORE_NDARRAY_NDARRAY_APPLY_BROADCAST_UNARY_H_

#include "oneflow/core/ndarray/ndarray_apply_broadcast_unary_core.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

template<DeviceType device_type, typename T, int NDIMS, const T (*unary_func)(const T)>
struct NdarrayApplyBroadcastUnary final {
  static void Apply(DeviceCtx* ctx, const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& x) {
    CheckBroadcastable(y, x);
    NdarrayApplyBroadcastUnaryCoreWrapper<device_type, T, NDIMS, unary_func>::Apply(ctx, y, x);
  }

  static void CheckBroadcastable(const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& x) {
    CHECK_EQ(y.shape().NumAxes(), x.shape().NumAxes());
    for (int i = 0; i < y.shape().NumAxes(); ++i) {
      CHECK(x.shape().At(i) == 1 || x.shape().At(i) == y.shape().At(i));
    }
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_NDARRAY_APPLY_BROADCAST_UNARY_H_
