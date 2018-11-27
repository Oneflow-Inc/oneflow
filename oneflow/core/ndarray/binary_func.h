#ifndef ONEFLOW_CORE_NDARRAY_BINARY_FUNC_H_
#define ONEFLOW_CORE_NDARRAY_BINARY_FUNC_H_

#include <cstdint>
#include <climits>
#include <cfloat>
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/common/util.h"

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

template<typename T>
OF_DEVICE_FUNC const T BinaryFuncMax(const T x, const T y) {
  return x > y ? x : y;
}

template<typename T>
OF_DEVICE_FUNC const T BinaryFuncMin(const T x, const T y) {
  return x < y ? x : y;
}

#define ARITHMETIC_BINARY_FUNC_SEQ    \
  OF_PP_MAKE_TUPLE_SEQ(BinaryFuncAdd) \
  OF_PP_MAKE_TUPLE_SEQ(BinaryFuncSub) \
  OF_PP_MAKE_TUPLE_SEQ(BinaryFuncMul) \
  OF_PP_MAKE_TUPLE_SEQ(BinaryFuncDiv)

template<typename T, const T (*binary_func)(const T, const T), typename Enable = void>
struct UnitOfBinaryFunc;

#define SPECIALIZE_UNIT_OF_BINARY_FUNC(binary_func, val_trait)                               \
  template<typename T, const T (*bfunc)(const T, const T)>                                   \
  struct UnitOfBinaryFunc<T, bfunc, typename std::enable_if<bfunc == &binary_func<T>>::type> \
      final {                                                                                \
    constexpr static T value = val_trait<T>::value;                                          \
  };
SPECIALIZE_UNIT_OF_BINARY_FUNC(BinaryFuncAdd, ZeroVal);
SPECIALIZE_UNIT_OF_BINARY_FUNC(BinaryFuncMul, OneVal);
SPECIALIZE_UNIT_OF_BINARY_FUNC(BinaryFuncMax, MinValue);
SPECIALIZE_UNIT_OF_BINARY_FUNC(BinaryFuncMin, MaxValue);
#undef SPECIALIZE_UNIT_OF_BINARY_FUNC

#define REDUCE_BINARY_FUNC_SEQ        \
  OF_PP_MAKE_TUPLE_SEQ(BinaryFuncAdd) \
  OF_PP_MAKE_TUPLE_SEQ(BinaryFuncMax) \
  OF_PP_MAKE_TUPLE_SEQ(BinaryFuncMin)

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_BINARY_FUNC_H_
