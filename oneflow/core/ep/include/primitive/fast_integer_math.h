/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_CORE_EP_PRIMITIVE_FAST_INTEGER_MATH_H_
#define ONEFLOW_CORE_EP_PRIMITIVE_FAST_INTEGER_MATH_H_
#include "oneflow/core/common/data_type.h"
#include <cassert>

namespace oneflow {

/*
  Copyright microsoft/onnxruntime
  https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cuda/shared_inc/fast_divmod.h
*/
template<typename T>
struct FastIntegerMath {
  OF_DEVICE_FUNC FastIntegerMath() {}
  OF_DEVICE_FUNC explicit FastIntegerMath(T operand) {
#if defined(__CUDA_ARCH__)
    int leading_zeroes = __clzll(operand);
#else
    int leading_zeroes = __builtin_clz(operand);
#endif
    bool is_power_2 = ((operand & (operand - 1)) == 0);
    if (is_power_2) {
      log2_operand_ = 31 - leading_zeroes;
    } else {
      log2_operand_ = -1;  // Set as flag.
    }
    operand_ = operand == 0 ? 1 : operand;
    assert(operand_ >= 1 && operand_ <= GetMaxVal<T>());
  }

  OF_DEVICE_FUNC T divides(T n) const {
    if (log2_operand_ >= 0) {
      return n >> log2_operand_;
    } else {
      return n / operand_;
    }
  }

  OF_DEVICE_FUNC T mod(T n) const { return n - divides(n) * operand_; }
  OF_DEVICE_FUNC T mul(T n) const {
    if (log2_operand_ >= 0) {
      return n << log2_operand_;
    } else {
      return n * operand_;
    }
  }
  OF_DEVICE_FUNC T add(T n) const { return n + operand_; }
  OF_DEVICE_FUNC T sub(T n) const { return n - operand_; }
  OF_DEVICE_FUNC void divmod(T n, T* q, T* r) const {
    *q = divides(n);
    *r = n - *q * operand_;
  }

  T operand_;
  int32_t log2_operand_;
};

template<>
struct FastIntegerMath<int32_t> {
  OF_DEVICE_FUNC FastIntegerMath() {}

  OF_DEVICE_FUNC explicit FastIntegerMath(const int32_t operand) {
    operand_ = operand == 0 ? 1 : operand;
    assert(operand_ >= 1 && operand_ <= GetMaxVal<uint32_t>());
    for (l_ = 0; l_ < 32; l_++)
      if ((1U << l_) >= operand_) break;

    uint64_t one = 1;
    uint64_t m = ((one << 32) * ((one << l_) - operand_)) / operand_ + 1;
    M_ = static_cast<uint32_t>(m);
    assert(M_ > 0 && M_ == m);
  }

  OF_DEVICE_FUNC int32_t divides(const int32_t n) const {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    uint32_t t = __umulhi(M_, n);
    return (t + n) >> l_;
#else
    // Using uint64_t for t, then t + n won't overflow.
    uint64_t t = ((uint64_t)M_ * n) >> 32;
    return static_cast<int>((t + n) >> l_);
#endif
  }

  OF_DEVICE_FUNC int32_t mod(int32_t n) const { return n - divides(n) * operand_; }
  OF_DEVICE_FUNC int32_t mul(int32_t n) const { return n * operand_; }
  OF_DEVICE_FUNC int32_t add(int32_t n) const { return n + operand_; }
  OF_DEVICE_FUNC int32_t sub(int32_t n) const { return n - operand_; }
  OF_DEVICE_FUNC void divmod(int32_t n, int32_t* q, int32_t* r) const {
    *q = divides(n);
    *r = n - *q * operand_;
  }

  uint32_t operand_;
  uint32_t M_;  // m' in the paper.
  uint32_t l_;  // l_ = ceil(log2(d_))
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_EP_PRIMITIVE_FAST_INTEGER_MATH_H_
