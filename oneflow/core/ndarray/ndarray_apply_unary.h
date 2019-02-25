#ifndef ONEFLOW_CORE_NDARRAY_NDARRAY_APPLY_UNARY_H_
#define ONEFLOW_CORE_NDARRAY_NDARRAY_APPLY_UNARY_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/ndarray/ndarray_apply_unary_core.h"

namespace oneflow {

template<DeviceType device_type, typename T, const T (*unary_func)(const T)>
struct NdarrayApplyUnary final {
  static void ImplaceApply(DeviceCtx* ctx, const XpuVarNdarray<T>& y) {
    NdarrayApplyUnaryCoreWrapper<device_type, T, unary_func>::ImplaceApply(ctx, y);
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_NDARRAY_APPLY_UNARY_H_
