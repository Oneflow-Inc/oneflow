#ifndef ONEFLOW_CORE_NDARRAY_UNARY_FUNC_H_
#define ONEFLOW_CORE_NDARRAY_UNARY_FUNC_H_

#if defined(__CUDACC__)
#include <cuda_fp16.hpp>
#endif
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

#define ARITHMETIC_UNARY_FUNC_NAME_SEQ (Identity)(Negative)(Exp)

#define PREPEND_PREFIX_UNARY_FUNC(name) OF_PP_CAT(UnaryFunc, name)
#define ARITHMETIC_UNARY_FUNC_SEQ \
  OF_PP_SEQ_MAP(PREPEND_PREFIX_UNARY_FUNC, ARITHMETIC_UNARY_FUNC_NAME_SEQ)

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

template<typename T>
struct UnaryFuncExp final {
  static OF_DEVICE_FUNC const T Invoke(const T x) { return std::exp(x); }
};
SPECIALIZE_CONST_TYPE_UNARY_FUNC(UnaryFuncExp);

template<>
struct UnaryFuncExp<float16> final {
  static OF_DEVICE_FUNC const float16 Invoke(const float16 x) {
    return float16(std::exp(static_cast<float>(x)));
  }
};
#define NO_HALF_UTIL_FOUND         \
  printf("cuda arch must >= 530"); \
  assert(false);                   \
  return __float2half(0.0)

#if defined(__CUDACC__)
template<>
struct UnaryFuncNegative<half> final {
  static __device__ __forceinline__ const half Invoke(const half x) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
    return __hneg(x);
#else
    NO_HALF_UTIL_FOUND;
#endif
  }
};
template<>
struct UnaryFuncExp<half> final {
  static __device__ __forceinline__ const half Invoke(const half x) {
    return __float2half(std::exp(__half2float(x)));
  }
};
#endif

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_UNARY_FUNC_H_
