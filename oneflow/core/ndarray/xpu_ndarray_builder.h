#ifndef ONEFLOW_CORE_NDARRAY_XPU_NDARRAY_BUILDER_H_
#define ONEFLOW_CORE_NDARRAY_XPU_NDARRAY_BUILDER_H_

#include "oneflow/core/ndarray/xpu_var_ndarray.h"
#include "oneflow/core/ndarray/xpu_reduce_ndarray.h"
#include "oneflow/core/ndarray/xpu_binary_func_ndarray.h"
#include "oneflow/core/ndarray/xpu_unary_func_ndarray.h"
#include "oneflow/core/ndarray/xpu_broadcast_ndarray.h"

namespace oneflow {

template<typename T, int NDIMS>
class XpuNdArrayBuilder final {
 public:
  OF_DEVICE_FUNC XpuNdArrayBuilder() = default;
  OF_DEVICE_FUNC ~XpuNdArrayBuilder() = default;

  template<const T (*unary_func)(const T), typename X>
  OF_DEVICE_FUNC XpuUnaryFuncNdarray<T, unary_func, X> Apply(const X& x) {
    return XpuUnaryFuncNdarray<T, unary_func, X>(x);
  }
  template<const T (*binary_func)(const T, const T), typename A, typename B>
  OF_DEVICE_FUNC XpuBinaryFuncNdarray<T, binary_func, A, B> Apply(const A& a, const B& b) {
    return XpuBinaryFuncNdarray<T, binary_func, A, B>(a, b);
  }
  OF_DEVICE_FUNC XpuBroadcastNdarray<const T> Broadcast(const ExecShape& shape,
                                                        const XpuVarNdarray<const T>& x) {
    return XpuBroadcastNdarray<const T>(shape, x);
  }
  template<typename X>
  OF_DEVICE_FUNC XpuReduceNdarray<T, NDIMS, X> Reduce(const ExecShape& shape, const X& x,
                                                      XpuVarNdarray<T>* storage) {
    return XpuReduceNdarray<T, NDIMS, X>(shape, x, storage);
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_XPU_NDARRAY_BUILDER_H_
