#ifndef ONEFLOW_CORE_NDARRAY_NDARRAY_APPLY_BROADCAST_BINARY_H_
#define ONEFLOW_CORE_NDARRAY_NDARRAY_APPLY_BROADCAST_BINARY_H_

#include "oneflow/core/ndarray/ndarray_apply_broadcast_binary_core.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

template<DeviceType device_type, typename T, int NDIMS, const T (*binary_func)(const T, const T)>
struct NdArrayApplyBroadcastBinary final {
  static void Apply(DeviceCtx* ctx, const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& a,
                    const XpuVarNdarray<const T>& b) {
    using NdarrayAssign = XpuNdArrayAssign<device_type, T, NDIMS>;
    using BroadcastBinary =
        NdArrayApplyBroadcastBinaryCoreWrapper<device_type, T, NDIMS, binary_func>;
    CheckBroadcastable(y, a, b);
    return BroadcastBinary::Apply(ctx, y, a, b);
    if (a.shape() == y.shape()) {
      NdarrayAssign::Assign(ctx, y, a);
      BroadcastBinary::ImplaceApply(ctx, y, b);
    } else if (b.shape() == y.shape()) {
      NdarrayAssign::Assign(ctx, y, b);
      BroadcastBinary::ImplaceApply(ctx, y, a);
    } else {
      BroadcastBinary::Apply(ctx, y, a, b);
    }
  }

  static void CheckBroadcastable(const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& a,
                                 const XpuVarNdarray<const T>& b) {
    CHECK_EQ(y.shape().NumAxes(), a.shape().NumAxes());
    CHECK_EQ(y.shape().NumAxes(), b.shape().NumAxes());
    for (int i = 0; i < y.shape().NumAxes(); ++i) {
      CHECK_EQ(y.shape().At(i), std::max(a.shape().At(i), b.shape().At(i)));
      if (a.shape().At(i) != b.shape().At(i)) {
        CHECK(a.shape().At(i) == 1 || b.shape().At(i) == 1);
      }
    }
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_NDARRAY_APPLY_BROADCAST_BINARY_H_
