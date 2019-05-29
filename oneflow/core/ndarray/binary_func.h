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
struct BinaryFuncAdd final {
  static inline OF_DEVICE_FUNC const T Invoke(const T x, const T y) { return x + y; }
};

template<typename T>
struct BinaryFuncSub final {
  static inline OF_DEVICE_FUNC const T Invoke(const T x, const T y) { return x - y; }
};

template<typename T>
struct BinaryFuncMul final {
  static inline OF_DEVICE_FUNC const T Invoke(const T x, const T y) { return x * y; }
};

template<typename T>
struct BinaryFuncDiv final {
  static inline OF_DEVICE_FUNC const T Invoke(const T x, const T y) { return x / y; }
};

template<typename T>
struct BinaryFuncMax final {
  static inline OF_DEVICE_FUNC const T Invoke(const T x, const T y) { return x > y ? x : y; }
};

template<typename T>
struct BinaryFuncMin final {
  static inline OF_DEVICE_FUNC const T Invoke(const T x, const T y) { return x < y ? x : y; }
};

#if defined(__CUDA_ARCH__)

#define NO_HALF_UTIL_FOUND         \
  printf("cuda arch must >= 530"); \
  assert(false);                   \
  return __float2half(0.0)
template<>
struct BinaryFuncAdd<half> final {
  static inline OF_DEVICE_FUNC const half Invoke(const half x, const half y) {
    return __hadd(x, y);
  }
};
template<>
struct BinaryFuncSub<half> final {
  static inline OF_DEVICE_FUNC const half Invoke(const half x, const half y) {
#if __CUDA_ARCH__ >= 530
    return __hsub(x, y);
#else
    NO_HALF_UTIL_FOUND;
#endif
  }
};
template<>
struct BinaryFuncMul<half> final {
  static inline OF_DEVICE_FUNC const half Invoke(const half x, const half y) {
#if __CUDA_ARCH__ >= 530
    return __hmul(x, y);
#else
    NO_HALF_UTIL_FOUND;
#endif
  }
};
template<>
struct BinaryFuncDiv<half> final {
  static inline OF_DEVICE_FUNC const half Invoke(const half x, const half y) {
#if __CUDA_ARCH__ >= 530
    return __hdiv(x, y);
#else
    NO_HALF_UTIL_FOUND;
#endif
  }
};
template<>
struct BinaryFuncMax<half> final {
  static inline OF_DEVICE_FUNC const half Invoke(const half x, const half y) {
#if __CUDA_ARCH__ >= 530
    return __hgt(x, y) ? x : y;
#else
    NO_HALF_UTIL_FOUND;
#endif
  }
};
template<>
struct BinaryFuncMin<half> final {
  static inline OF_DEVICE_FUNC const half Invoke(const half x, const half y) {
#if __CUDA_ARCH__ >= 530
    return __hlt(x, y) ? x : y;
#else
    NO_HALF_UTIL_FOUND;
#endif
  }
};
#endif

#define ARITHMETIC_BINARY_FUNC_SEQ (BinaryFuncAdd)(BinaryFuncSub)(BinaryFuncMul)(BinaryFuncDiv)

template<typename T, template<typename> class binary_func>
struct UnitOfBinaryFunc;

#define SPECIALIZE_UNIT_OF_BINARY_FUNC(binary_func, get_val)      \
  template<typename T>                                            \
  struct UnitOfBinaryFunc<T, binary_func> final {                 \
    static inline OF_DEVICE_FUNC T Val() { return get_val<T>(); } \
  };
SPECIALIZE_UNIT_OF_BINARY_FUNC(BinaryFuncAdd, GetZeroVal);
SPECIALIZE_UNIT_OF_BINARY_FUNC(BinaryFuncMul, GetOneVal);
SPECIALIZE_UNIT_OF_BINARY_FUNC(BinaryFuncMax, GetMinVal);
SPECIALIZE_UNIT_OF_BINARY_FUNC(BinaryFuncMin, GetMaxVal);
#undef SPECIALIZE_UNIT_OF_BINARY_FUNC

#define REDUCE_BINARY_FUNC_SEQ (BinaryFuncAdd)(BinaryFuncMax)(BinaryFuncMin)

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_BINARY_FUNC_H_
