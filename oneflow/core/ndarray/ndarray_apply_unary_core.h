#ifndef ONEFLOW_CORE_NDARRAY_NDARRAY_APPLY_UNARY_CORE_H_
#define ONEFLOW_CORE_NDARRAY_NDARRAY_APPLY_UNARY_CORE_H_

#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/ndarray/xpu_unary_func_ndarray.h"
#include "oneflow/core/ndarray/xpu_var_ndarray.h"
#include "oneflow/core/ndarray/ndarray_util.h"

namespace oneflow {

template<DeviceType device_type, typename T, const T (*unary_func)(const T)>
struct NdArrayApplyUnaryCoreWrapper final {
  static void ImplaceApply(DeviceCtx* ctx, const XpuVarNdarray<T>& y);
};

template<typename T, const T (*unary_func)(const T)>
struct NdArrayApplyUnaryCore final {
  OF_DEVICE_FUNC static void ImplaceApply(T* y, size_t n) {
    XPU_1D_KERNEL_LOOP(i, n) { y[i] = unary_func(y[i]); }
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_NDARRAY_APPLY_UNARY_CORE_H_
