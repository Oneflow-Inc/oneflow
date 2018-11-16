#ifndef ONEFLOW_CORE_NDARRAY_BINARY_FUNC_H_
#define ONEFLOW_CORE_NDARRAY_BINARY_FUNC_H_

#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<typename T>
OF_DEVICE_FUNC const T BinaryFuncAdd(const T x, const T y) {
  return x + y;
}

template<typename T>
OF_DEVICE_FUNC const T BinaryFuncSub(const T x, const T y) {
  return x - y;
}

template<typename T>
OF_DEVICE_FUNC const T BinaryFuncMul(const T x, const T y) {
  return x * y;
}

template<typename T>
OF_DEVICE_FUNC const T BinaryFuncDiv(const T x, const T y) {
  return x / y;
}

#define ARITHMETIC_BINARY_FUNC_SEQ    \
  OF_PP_MAKE_TUPLE_SEQ(BinaryFuncAdd) \
  OF_PP_MAKE_TUPLE_SEQ(BinaryFuncSub) \
  OF_PP_MAKE_TUPLE_SEQ(BinaryFuncMul) \
  OF_PP_MAKE_TUPLE_SEQ(BinaryFuncDiv)

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_BINARY_FUNC_H_
