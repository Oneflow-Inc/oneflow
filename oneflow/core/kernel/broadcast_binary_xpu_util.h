#ifndef ONEFLOW_CORE_KERNEL_BROADCAST_BINARY_XPU_UTIL_H_
#define ONEFLOW_CORE_KERNEL_BROADCAST_BINARY_XPU_UTIL_H_

#include "oneflow/core/ndarray/xpu_var_ndarray.h"
#include "oneflow/core/ndarray/xpu_broadcast_ndarray.h"
#include "oneflow/core/ndarray/xpu_binary_func_ndarray.h"

namespace oneflow {

template<typename T, int NDIMS, const T (*binary_func)(const T, const T)>
struct BroadcastBinaryXpuUtil final {
  OF_DEVICE_FUNC static void Forward(XpuVarNdarray<T>* y, const XpuVarNdarray<const T>& a,
                                     const XpuVarNdarray<const T>& b) {
    XpuBroadcastNdarray<const T> a_broadcasted(y->shape(), a);
    XpuBroadcastNdarray<const T> b_broadcasted(y->shape(), b);
    XpuBinaryFuncNdarray<const T, binary_func> binary_func_result(a_broadcasted, b_broadcasted);
    y->template Assign<NDIMS>(binary_func_result);
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_BROADCAST_BINARY_XPU_UTIL_H_
