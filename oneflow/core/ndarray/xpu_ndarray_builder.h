#ifndef ONEFLOW_CORE_NDARRAY_XPU_NDARRAY_BUILDER_H_
#define ONEFLOW_CORE_NDARRAY_XPU_NDARRAY_BUILDER_H_

#include "oneflow/core/ndarray/xpu_var_ndarray.h"
#include "oneflow/core/ndarray/xpu_binary_func_ndarray.h"
#include "oneflow/core/ndarray/xpu_unary_func_ndarray.h"
#include "oneflow/core/ndarray/xpu_broadcast_ndarray.h"
#include "oneflow/core/ndarray/xpu_transpose_ndarray.h"
#include "oneflow/core/ndarray/xpu_reshape_ndarray.h"

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
  OF_DEVICE_FUNC XpuBroadcastNdarray<const T> Broadcast(const XpuShape& shape,
                                                        const XpuVarNdarray<const T>& x) {
    return XpuBroadcastNdarray<const T>(shape, x);
  }
  template<typename X>
  OF_DEVICE_FUNC XpuTransposeNdarray<T, NDIMS, X> Transpose(const X& x, const int64_t perm[NDIMS]) {
    return XpuTransposeNdarray<T, NDIMS, X>(x, perm);
  }
  template<int ndims = NDIMS, typename X>
  OF_DEVICE_FUNC XpuReshapeNdarray<T, ndims, X> Reshape(const X& x, const int64_t dim[ndims]) {
    return XpuReshapeNdarray<T, ndims, X>(x, dim);
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_XPU_NDARRAY_BUILDER_H_
