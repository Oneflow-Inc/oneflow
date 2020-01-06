#ifndef ONEFLOW_CORE_KERNEL_UTIL_CUDA_HALF_UTIL_H_
#define ONEFLOW_CORE_KERNEL_UTIL_CUDA_HALF_UTIL_H_

#include "oneflow/core/device/cuda_util.h"

namespace oneflow {

#define HALF_CHECK_FAILED                                             \
  printf("half operations are only supported when CUDA_ARCH >= 530"); \
  assert(false)

__inline__ __device__ half hone() { return __float2half(1.0); }
__inline__ __device__ half hzero() { return __float2half(0.0); }

__inline__ half float16_2half(float16 x) {
  // TODO: Potential loss of accuracy
  half* ret = reinterpret_cast<half*>(&x);
  return *ret;
}

__inline__ float16 half2float16(half x) {
  // TODO: Potential loss of accuracy
  float16* ret = reinterpret_cast<float16*>(&x);
  return *ret;
}
}

#endif  // ONEFLOW_CORE_KERNEL_UTIL_CUDA_HALF_UTIL_H_
