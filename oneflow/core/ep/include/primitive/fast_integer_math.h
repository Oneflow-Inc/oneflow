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
  Copyright NVIDIA/DALI
  https://github.com/NVIDIA/DALI/blob/main/include/dali/core/fast_div.h
*/
template<typename component>
struct lohi {
  component lo, hi;
};

template<typename T>
OF_DEVICE_FUNC lohi<T> operator<<(lohi<T> x, unsigned sh) {
  using U = typename std::make_unsigned<T>::type;
  static constexpr unsigned bits = sizeof(T) * 8;
  if (sh == 0) {
    return x;
  } else if (sh >= bits) {
    return {0, x.lo << (sh - bits)};
  } else {
    return {x.lo << sh, x.hi << sh | U(x.lo) >> (bits - sh)};
  }
}

template<typename T>
OF_DEVICE_FUNC lohi<T> operator>>(lohi<T> x, unsigned sh) {
  using U = typename std::make_unsigned<T>::type;
  static constexpr unsigned bits = sizeof(T) * 8;
  if (sh == 0) {
    return x;
  } else if (sh >= bits) {
    return {x.hi >> (sh - bits), 0};
  } else {
    return {U(x.lo) >> sh | x.hi << (bits - sh), x.hi >> sh};
  }
}

template<typename T>
OF_DEVICE_FUNC lohi<T>& operator<<=(lohi<T>& x, unsigned sh) {
  x = x << sh;
  return x;
}

template<typename T>
OF_DEVICE_FUNC lohi<T>& operator>>=(lohi<T>& x, unsigned sh) {
  x = x >> sh;
  return x;
}

template<typename T>
OF_DEVICE_FUNC lohi<T> operator-(lohi<T> a, lohi<T> b) {
  lohi<T> ret;
  ret.lo = a.lo - b.lo;
  int borrow = (b.lo && ret.lo > a.lo);
  ret.hi = a.hi - b.hi - borrow;
  return ret;
}

OF_DEVICE_FUNC uint32_t div_lohi(uint32_t lo, uint32_t hi, uint32_t operand) {
  return (static_cast<uint64_t>(hi) << 32 | lo) / operand;
}

OF_DEVICE_FUNC lohi<uint64_t> mull(uint64_t a, uint64_t b) {
  lohi<uint64_t> ret;
#ifdef __CUDA_ARCH__
  ret.lo = a * b;
  ret.hi = __umul64hi(a, b);
#else
  unsigned __int128 m = (unsigned __int128)a * b;
  ret.lo = m;
  ret.hi = m >> 64;
#endif
  return ret;
}

template<typename T>
OF_DEVICE_FUNC int ilog2(T x) noexcept {
  int n = 0;
  while (x >>= 1) n++;
  return n;
};

OF_DEVICE_FUNC uint64_t div_lohi(uint64_t lo, uint64_t hi, uint64_t operand) {
#if defined(__x86_64) && defined(__GNUC__) && !defined(__CUDA_ARCH__)
  // I hope this gets compiled to dividing rdx:rax register pair by a 64-bit value
  return (static_cast<unsigned __int128>(hi) << 64 | lo) / operand;
#else
#if CUDA_VERSION >= 11500
  // NVCC support __int128 in device code in CUDA11.5 (zhengzekang)
  return (static_cast<unsigned __int128>(hi) << 64 | lo) / operand;
#else
  // NVCC doesn't support __int128 in device code, so we need a bit of hackery
  if (!hi)  // No high part? Just divide in 64-bits and be done.
    return lo / operand;

  // long division:

  int lnum = ilog2(hi) + 64;
  int lden = ilog2(operand);

  lohi<uint64_t> num = {lo, hi};
  lohi<uint64_t> den = {operand, 0};

  // calculate MSB positions...
  int sh = lnum - lden;
  // .. and align numerator and denominator
  den <<= sh;

  uint64_t q = 0;

  while (sh >= 0) {
    lohi<uint64_t> dif = num - den;  // this serves both as difference and comparison
    if (static_cast<int64_t>(dif.hi) >= 0) {
      num = dif;
      q |= static_cast<uint64_t>(1) << sh;
    }
    sh--;
    den >>= 1;
  }
  return q;
#endif
#endif
}

// fast_div works only with unsigned integers. (zhengzekang)
template<typename T>
struct FastIntegerMath {
  uint64_t operand_;
  uint64_t mul_factor_;
  uint8_t add_;
  uint8_t shift_;

  OF_DEVICE_FUNC FastIntegerMath() {}

  OF_DEVICE_FUNC FastIntegerMath(T operand) { init(operand); }

  OF_DEVICE_FUNC void init(T operand) {
    this->operand_ = static_cast<uint64_t>(operand);
    this->mul_factor_ = 1;
    this->shift_ = 0;
    this->add_ = 0;
    if (operand == 0) { return; }

    int log_div = ilog2(operand);
    this->shift_ = log_div;

    if ((operand & (operand - 1)) == 0) {
      this->mul_factor_ = 0;
      return;
    }

    uint64_t m_lo = div_lohi(0, uint64_t(1) << log_div, operand);
    uint64_t m_hi = div_lohi(uint64_t(1) << log_div, uint64_t(1) << log_div, operand);
    this->add_ = (m_lo == m_hi) ? 1 : 0;  // round-up failed, use round-down method
    this->mul_factor_ = m_hi;
  }

  OF_DEVICE_FUNC T divides(T x) const {
    // If the operand is a power of 2, the multiplier would be 2^64, which is out of range
    // - therefore, powers of 2 get special treatment and the multiplication is skipped.
    x = static_cast<uint64_t>(x);
#ifdef __CUDA_ARCH__
    if (mul_factor_) {
      x = __umul64hi(x + add_, mul_factor_);
    }
    return x >> shift_;
#else
    if (mul_factor_) {
      uint64_t hi = static_cast<unsigned __int128>(x + add_) * mul_factor_ >> 64;
      return hi >> shift_;
    } else {
      return x >> shift_;
    }
#endif
  }

  OF_DEVICE_FUNC T mod(T n) const { return n - divides(n) * operand_; }
  OF_DEVICE_FUNC T mul(T n) const { return n * operand_; }
  OF_DEVICE_FUNC T add(T n) const { return n + operand_; }
  OF_DEVICE_FUNC T sub(T n) const { return n - operand_; }
  OF_DEVICE_FUNC void divmod(T n, T* q, T* r) const {
    *q = divides(n);
    *r = n - *q * operand_;
  }
};

template<>
struct FastIntegerMath<int32_t> {
  uint32_t operand_;
  uint32_t mul_factor_;
  uint8_t add_;
  uint8_t shift_;

  OF_DEVICE_FUNC FastIntegerMath() {}

  OF_DEVICE_FUNC FastIntegerMath(int32_t operand) { init(operand); }

  OF_DEVICE_FUNC void init(int32_t operand) {
    this->operand_ = static_cast<uint32_t>(operand);
    this->mul_factor_ = 1;
    this->shift_ = 0;
    this->add_ = 0;
    if (operand == 0) { return; }

    int log_div = ilog2(operand);
    this->shift_ = log_div;

    if ((operand & (operand - 1)) == 0) {
      this->mul_factor_ = 0;
      return;
    }

    uint32_t m_lo = div_lohi(0, uint32_t(1) << log_div, operand);
    uint32_t m_hi = div_lohi(uint32_t(1) << log_div, uint32_t(1) << log_div, operand);
    this->add_ = (m_lo == m_hi) ? 1 : 0;  // round-up failed, use round-down method
    this->mul_factor_ = m_hi;
  }

  OF_DEVICE_FUNC int32_t divides(int32_t x) const {
#ifdef __CUDA_ARCH__
    if (mul_factor_) { x = __umulhi(x + add_, mul_factor_); }
    return x >> shift_;
#else
    if (mul_factor_) {
      uint32_t hi = static_cast<uint64_t>(x + add_) * mul_factor_ >> 32;
      return hi >> shift_;
    } else {
      return x >> shift_;
    }
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
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_EP_PRIMITIVE_FAST_INTEGER_MATH_H_
