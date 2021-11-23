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
#ifndef ONEFLOW_CORE_COMMON_FAST_DIV_MOD_H_
#define ONEFLOW_CORE_COMMON_FAST_DIV_MOD_H_
#include "oneflow/core/common/data_type.h"
#include <cassert>

namespace oneflow {

/*
  Copyright microsoft/onnxruntime
  https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cuda/shared_inc/fast_divmod.h
*/
template<typename T>
struct FastDivMod {
  OF_DEVICE_FUNC explicit FastDivMod(const int divisor);
  OF_DEVICE_FUNC int32_t div(const int32_t n);
  OF_DEVICE_FUNC int32_t mod(const int32_t n);
  OF_DEVICE_FUNC void divmod(const int n, int& q, int& r);
  uint32_t divisor_;
  int32_t log2_divisor;
  bool is_power_2;
  uint32_t M_;  // m' in the paper.
  uint32_t l_;  // l_ = ceil(log2(d_))
};

template<>
struct FastDivMod<int32_t> {
  OF_DEVICE_FUNC explicit FastDivMod(const int32_t divisor = 1) {
    assert(divisor >= 1);
#if defined(__CUDA_ARCH__)
    int leading_zeroes = __clz(divisor);
#else
    int leading_zeroes = __builtin_clz(divisor);
#endif
    log2_divisor = 31 - leading_zeroes;
    is_power_2 = ((divisor & (divisor - 1)) == 0);
    divisor_ = divisor == 0 ? 1 : divisor;
    assert(divisor_ >= 1 && divisor_ <= GetMaxVal<uint32_t>());
    for (l_ = 0; l_ < 32; l_++)
      if ((1U << l_) >= divisor_) break;

    uint64_t one = 1;
    uint64_t m = ((one << 32) * ((one << l_) - divisor_)) / divisor_ + 1;
    M_ = static_cast<uint32_t>(m);
    assert(M_ > 0 && M_ == m);
  }

  OF_DEVICE_FUNC int32_t div(const int32_t n) const {
    if (is_power_2) {
      return n >> log2_divisor;
    } else {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      uint32_t t = __umulhi(M_, n);
      return (t + n) >> l_;
#else
      // Using uint64_t for t, then t + n won't overflow.
      uint64_t t = ((uint64_t)M_ * n) >> 32;
      return static_cast<int>((t + n) >> l_);
#endif
    }
  }

  OF_DEVICE_FUNC int32_t mod(const int32_t n) const { return n - div(n) * divisor_; }

  OF_DEVICE_FUNC void divmod(const int32_t n, int32_t& q, int32_t& r) const {
    q = div(n);
    r = n - q * divisor_;
  }
  uint32_t divisor_;
  int32_t log2_divisor;
  bool is_power_2;
  uint32_t M_;  // m' in the paper.
  uint32_t l_;  // l_ = ceil(log2(d_))
};

template<>
struct FastDivMod<int64_t> {
  OF_DEVICE_FUNC explicit FastDivMod(const int64_t divisor = 1) {
    assert(divisor >= 1);
#if defined(__CUDA_ARCH__)
    int leading_zeroes = __clzll(divisor);
#else
    int leading_zeroes = __builtin_clz(divisor);
#endif
    log2_divisor = 31 - leading_zeroes;
    is_power_2 = ((divisor & (divisor - 1)) == 0);
    divisor_ = divisor == 0 ? 1 : divisor;
    assert(divisor_ >= 1 && divisor_ <= GetMaxVal<uint64_t>());
    // Not used in int64_t division.
    M_ = 0;
    l_ = 0;
  }

  OF_DEVICE_FUNC int64_t div(const int64_t n) const {
    if (is_power_2) {
      return n >> log2_divisor;
    } else {
      return n / divisor_;
    }
  }

  OF_DEVICE_FUNC int64_t mod(const int64_t n) const { return n - div(n) * divisor_; }

  OF_DEVICE_FUNC void divmod(const int64_t n, int64_t& q, int64_t& r) const {
    q = div(n);
    r = n - q * divisor_;
  }
  uint32_t divisor_;  // divisor
  int32_t log2_divisor;
  bool is_power_2;
  uint32_t M_;  // m' in the paper.
  uint32_t l_;  // l_ = ceil(log2(d_))
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_FAST_DIV_MOD_H_
