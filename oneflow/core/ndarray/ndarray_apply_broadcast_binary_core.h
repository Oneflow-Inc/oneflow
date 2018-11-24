#ifndef ONEFLOW_CORE_NDARRAY_NDARRAY_APPLY_BROADCAST_BINARY_CORE_H_
#define ONEFLOW_CORE_NDARRAY_NDARRAY_APPLY_BROADCAST_BINARY_CORE_H_

#include "oneflow/core/ndarray/ndarray_util.h"
#include "oneflow/core/ndarray/xpu_ndarray_builder.h"
#include "oneflow/core/ndarray/xpu_var_ndarray.h"
#include "oneflow/core/ndarray/binary_func.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T, int NDIMS, const T (*binary_func)(const T, const T)>
struct NdArrayApplyBroadcastBinaryCoreWrapper final {
  static void Apply(DeviceCtx* ctx, const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& a,
                    const XpuVarNdarray<const T>& b);
  static void ImplaceApply(DeviceCtx* ctx, const XpuVarNdarray<T>& y,
                           const XpuVarNdarray<const T>& x);
};

template<typename T, int NDIMS, const T (*binary_func)(const T, const T)>
struct NdArrayApplyBroadcastBinaryCore final {
  OF_DEVICE_FUNC static void Apply(const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& a,
                                   const XpuVarNdarray<const T>& b) {
    XpuNdArrayBuilder<T, NDIMS> ndarray;
    const auto& a_broadcasted = ndarray.Broadcast(y.shape(), a);
    const auto& b_broadcasted = ndarray.Broadcast(y.shape(), b);
    const auto& ret = ndarray.template Apply<binary_func>(a_broadcasted, b_broadcasted);
    y.template Assign<NDIMS>(ret);
  }
  OF_DEVICE_FUNC static void ImplaceApply(const XpuVarNdarray<T>& y,
                                          const XpuVarNdarray<const T>& x) {
    XpuNdArrayBuilder<T, NDIMS> ndarray;
    const auto& x_broadcasted = ndarray.Broadcast(y.shape(), x);
    y.template BinaryAssign<binary_func, NDIMS>(x_broadcasted);
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_NDARRAY_APPLY_BROADCAST_BINARY_CORE_H_
