#ifndef ONEFLOW_CORE_NDARRAY_UNARY_FUNC_H_
#define ONEFLOW_CORE_NDARRAY_UNARY_FUNC_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

#define ARITHMETIC_UNARY_FUNC_SEQ         \
  OF_PP_MAKE_TUPLE_SEQ(UnaryFuncIdentity) \
  OF_PP_MAKE_TUPLE_SEQ(UnaryFuncMinus)

template<typename T>
OF_DEVICE_FUNC const T UnaryFuncIdentity(const T x) {
  return x;
}

template<typename T>
OF_DEVICE_FUNC const T UnaryFuncMinus(const T x) {
  return -x;
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_UNARY_FUNC_H_
