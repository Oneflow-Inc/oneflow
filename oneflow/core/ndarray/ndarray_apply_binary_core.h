#ifndef ONEFLOW_CORE_NDARRAY_NDARRAY_APPLY_BINARY_CORE_H_
#define ONEFLOW_CORE_NDARRAY_NDARRAY_APPLY_BINARY_CORE_H_

#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/ndarray/xpu_binary_func_ndarray.h"
#include "oneflow/core/ndarray/xpu_var_ndarray.h"
#include "oneflow/core/ndarray/xpu_util.h"
#include "oneflow/core/ndarray/binary_func.h"

namespace oneflow {

template<DeviceType device_type, typename T, template<typename> class binary_func>
struct NdarrayApplyBinaryCoreWrapper final {
  static void Apply(DeviceCtx* ctx,
                    const XpuVarNdarray<typename BinaryFuncTrait<binary_func, T>::return_type>& y,
                    const XpuVarNdarray<const T>& a, const XpuVarNdarray<const T>& b);
  static void InplaceApply(DeviceCtx* ctx, const XpuVarNdarray<T>& y,
                           const XpuVarNdarray<const T>& x);
};

template<typename T, template<typename> class binary_func>
struct NdarrayApplyBinaryCore final {
  OF_DEVICE_FUNC static void Apply(size_t n,
                                   typename BinaryFuncTrait<binary_func, T>::return_type* y,
                                   const T* a, const T* b) {
    XPU_1D_KERNEL_LOOP(i, n) { y[i] = binary_func<T>::Invoke(a[i], b[i]); }
  }
  OF_DEVICE_FUNC static void InplaceApply(size_t n, T* y, const T* x) {
    XPU_1D_KERNEL_LOOP(i, n) { y[i] = binary_func<T>::Invoke(y[i], x[i]); }
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_NDARRAY_APPLY_BINARY_CORE_H_
