#ifndef ONEFLOW_CORE_NDARRAY_BINARY_FUNC_H_
#define ONEFLOW_CORE_NDARRAY_BINARY_FUNC_H_

#include <cstdint>
#include <climits>
#include <cfloat>

#if defined(__CUDACC__)
#include <cuda_fp16.h>
#endif

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

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
template<>
inline OF_DEVICE_FUNC const half BinaryFuncAdd<half>(const half x, const half y) {
  return __hadd(x, y);
}
template<>
inline OF_DEVICE_FUNC const half BinaryFuncSub<half>(const half x, const half y) {
  return __hsub(x, y);
}
template<>
inline OF_DEVICE_FUNC const half BinaryFuncMul<half>(const half x, const half y) {
  return __hmul(x, y);
}
template<>
inline OF_DEVICE_FUNC const half BinaryFuncDiv<half>(const half x, const half y) {
  return __hdiv(x, y);
}
template<>
inline OF_DEVICE_FUNC const half BinaryFuncMax<half>(const half x, const half y) {
  return __hgt(x, y) ? x : y;
}
template<>
inline OF_DEVICE_FUNC const half BinaryFuncMin<half>(const half x, const half y) {
  return __hlt(x, y) ? x : y;
}
#endif

#define ARITHMETIC_BINARY_FUNC_SEQ    \
  OF_PP_MAKE_TUPLE_SEQ(BinaryFuncAdd) \
  OF_PP_MAKE_TUPLE_SEQ(BinaryFuncSub) \
  OF_PP_MAKE_TUPLE_SEQ(BinaryFuncMul) \
  OF_PP_MAKE_TUPLE_SEQ(BinaryFuncDiv)

template<typename T, const T (*binary_func)(const T, const T), typename Enable = void>
struct UnitOfBinaryFunc;

#define SPECIALIZE_UNIT_OF_BINARY_FUNC(binary_func, get_val)                                 \
  template<typename T, const T (*bfunc)(const T, const T)>                                   \
  struct UnitOfBinaryFunc<T, bfunc, typename std::enable_if<bfunc == &binary_func<T>>::type> \
      final {                                                                                \
    static OF_DEVICE_FUNC T Val() { return get_val<T>(); }                                   \
  };
SPECIALIZE_UNIT_OF_BINARY_FUNC(BinaryFuncAdd, GetZeroVal);
SPECIALIZE_UNIT_OF_BINARY_FUNC(BinaryFuncMul, GetOneVal);
SPECIALIZE_UNIT_OF_BINARY_FUNC(BinaryFuncMax, GetMinVal);
SPECIALIZE_UNIT_OF_BINARY_FUNC(BinaryFuncMin, GetMaxVal);
#undef SPECIALIZE_UNIT_OF_BINARY_FUNC

#define REDUCE_BINARY_FUNC_SEQ        \
  OF_PP_MAKE_TUPLE_SEQ(BinaryFuncAdd) \
  OF_PP_MAKE_TUPLE_SEQ(BinaryFuncMax) \
  OF_PP_MAKE_TUPLE_SEQ(BinaryFuncMin)

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_BINARY_FUNC_H_
