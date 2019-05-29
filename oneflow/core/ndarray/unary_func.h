#ifndef ONEFLOW_CORE_NDARRAY_UNARY_FUNC_H_
#define ONEFLOW_CORE_NDARRAY_UNARY_FUNC_H_

#if defined(__CUDACC__)
#include <cuda_fp16.hpp>
#endif
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

#define ARITHMETIC_UNARY_FUNC_SEQ (UnaryFuncIdentity)(UnaryFuncNegative)

#define SPECIALIZE_CONST_TYPE_UNARY_FUNC(func_struct)                                     \
  template<typename T>                                                                    \
  struct func_struct<const T> final {                                                     \
    static OF_DEVICE_FUNC const T Invoke(const T x) { return func_struct<T>::Invoke(x); } \
  }

template<typename T>
struct UnaryFuncIdentity final {
  static OF_DEVICE_FUNC const T Invoke(const T x) { return x; }
};

template<typename T>
struct UnaryFuncNegative final {
  static OF_DEVICE_FUNC const T Invoke(const T x) { return -x; }
};
SPECIALIZE_CONST_TYPE_UNARY_FUNC(UnaryFuncNegative);

#define NO_HALF_UTIL_FOUND         \
  printf("cuda arch must >= 530"); \
  assert(false);                   \
  return __float2half(0.0)

#if defined(__CUDA_ARCH__)
template<>
struct UnaryFuncNegative<half> final {
  static OF_DEVICE_FUNC const half Invoke(const half x) {
#if __CUDA_ARCH__ >= 530
    return __hneg(x);
#else
    NO_HALF_UTIL_FOUND;
#endif
  }
};
#endif

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_UNARY_FUNC_H_
