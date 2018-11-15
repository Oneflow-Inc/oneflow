#ifndef ONEFLOW_CORE_KERNEL_BROADCAST_BINARY_XPU_UTIL_H_
#define ONEFLOW_CORE_KERNEL_BROADCAST_BINARY_XPU_UTIL_H_

#include "oneflow/core/ndarray/xpu_ndarray_builder.h"

namespace oneflow {

template<typename T, int NDIMS, const T (*binary_func)(const T, const T)>
struct BroadcastBinaryXpuUtil final {
  OF_DEVICE_FUNC static void Forward(XpuVarNdarray<T>* y, const XpuVarNdarray<const T>& a,
                                     const XpuVarNdarray<const T>& b) {
    XpuNdArrayBuilder<T, NDIMS> ndarray;
    const auto& a_broadcasted = ndarray.Broadcast(y->shape(), a);
    const auto& b_broadcasted = ndarray.Broadcast(y->shape(), b);
    const auto& ret = ndarray.template Apply<binary_func>(a_broadcasted, b_broadcasted);
    y->template Assign<NDIMS>(ret);
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_BROADCAST_BINARY_XPU_UTIL_H_
