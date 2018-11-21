#ifndef ONEFLOW_CORE_NDARRAY_UNARY_FUNC_H_
#define ONEFLOW_CORE_NDARRAY_UNARY_FUNC_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

#define ARITHMETIC_UNARY_FUNC_SEQ         \
  OF_PP_MAKE_TUPLE_SEQ(UnaryFuncIdentity) \
  OF_PP_MAKE_TUPLE_SEQ(UnaryFuncMinus)    \
  OF_PP_MAKE_TUPLE_SEQ(UnaryFuncLog2)     \
  OF_PP_MAKE_TUPLE_SEQ(UnaryFuncExp2)

template<typename T>
OF_DEVICE_FUNC const T UnaryFuncIdentity(const T x) {
  return x;
}

template<typename T>
OF_DEVICE_FUNC const T UnaryFuncMinus(const T x) {
  return -x;
}

template<typename T>
OF_DEVICE_FUNC
    typename std::enable_if<std::is_same<T, float>::value || std::is_same<T, double>::value,
                            const T>::type
    UnaryFuncLog2(const T x) {
#if defined(__CUDACC__)
  return log2(x);
#else
  return std::log2(x);
#endif
}

template<typename T>
OF_DEVICE_FUNC
    typename std::enable_if<!(std::is_same<T, float>::value || std::is_same<T, double>::value),
                            const T>::type
    UnaryFuncLog2(const T x) {
#if defined(__CUDACC__)
  return log2(static_cast<float>(x));
#else
  return std::log2(x);
#endif
}

template<typename T>
OF_DEVICE_FUNC
    typename std::enable_if<std::is_same<T, float>::value || std::is_same<T, double>::value,
                            const T>::type
    UnaryFuncExp2(const T x) {
#if defined(__CUDACC__)
  return exp2(x);
#else
  return std::exp2(x);
#endif
}

template<typename T>
OF_DEVICE_FUNC
    typename std::enable_if<!(std::is_same<T, float>::value || std::is_same<T, double>::value),
                            const T>::type
    UnaryFuncExp2(const T x) {
#if defined(__CUDACC__)
  return exp2(static_cast<float>(x));
#else
  return std::exp2(x);
#endif
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_UNARY_FUNC_H_
