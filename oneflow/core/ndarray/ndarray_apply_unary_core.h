#ifndef ONEFLOW_CORE_NDARRAY_NDARRAY_APPLY_UNARY_CORE_H_
#define ONEFLOW_CORE_NDARRAY_NDARRAY_APPLY_UNARY_CORE_H_

#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/ndarray/xpu_unary_func_ndarray.h"
#include "oneflow/core/ndarray/xpu_var_ndarray.h"
#include "oneflow/core/ndarray/xpu_util.h"

namespace oneflow {

template<DeviceType device_type, typename T, template<typename> class unary_func>
struct NdarrayApplyUnaryCoreWrapper final {
  static void ImplaceApply(DeviceCtx* ctx, const XpuVarNdarray<T>& y);
};

template<typename T, template<typename> class unary_func>
struct NdarrayApplyUnaryCore final {
  OF_DEVICE_FUNC static void ImplaceApply(T* y, size_t n) {
    XPU_1D_KERNEL_LOOP(i, n) { y[i] = unary_func<T>::Invoke(y[i]); }
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_NDARRAY_APPLY_UNARY_CORE_H_
