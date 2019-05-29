#ifndef ONEFLOW_CORE_NDARRAY_UNARY_FUNC_H_
#define ONEFLOW_CORE_NDARRAY_UNARY_FUNC_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

#define ARITHMETIC_UNARY_FUNC_SEQ (UnaryFuncIdentity)(UnaryFuncMinus)

template<typename T>
struct UnaryFuncIdentity final {
  static OF_DEVICE_FUNC const T Invoke(const T x) { return x; }
};

template<typename T>
struct UnaryFuncMinus final {
  static OF_DEVICE_FUNC const T Invoke(const T x) { return -x; }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_UNARY_FUNC_H_
