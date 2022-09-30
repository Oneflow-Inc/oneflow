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
  uint32_t, uint64_t fast divide algorithem is from NVIDIA/DALI
  https://github.com/NVIDIA/DALI/blob/main/include/dali/core/fast_div.h

  int32_t, int64_t fast divide algorithem is from libdivide
  https://github.com/ridiculousfish/libdivide
*/
namespace {

// DALI function
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

// libdivide function
enum {
  LIBDIVIDE_16_SHIFT_MASK = 0x1F,
  LIBDIVIDE_32_SHIFT_MASK = 0x1F,
  LIBDIVIDE_64_SHIFT_MASK = 0x3F,
  LIBDIVIDE_ADD_MARKER = 0x40,
  LIBDIVIDE_NEGATIVE_DIVISOR = 0x80
};

OF_DEVICE_FUNC uint32_t libdivide_64_div_32_to_32(uint32_t u1, uint32_t u0, uint32_t v,
                                                  uint32_t* r) {
  uint64_t n = ((uint64_t)u1 << 32) | u0;
  uint32_t result = (uint32_t)(n / v);
  *r = (uint32_t)(n - result * (uint64_t)v);
  return result;
}

OF_DEVICE_FUNC int32_t static libdivide_mullhi_s32(int32_t x, int32_t y) {
  int64_t xl = x, yl = y;
  int64_t rl = xl * yl;
  // needs to be arithmetic shift
  return (int32_t)(rl >> 32);
}

OF_DEVICE_FUNC int32_t libdivide_count_leading_zeros64(uint64_t val) {
#if defined(__CUDA_ARCH__)
  return __clzll(val);
#else
  return __builtin_clzll(val);
#endif
}

OF_DEVICE_FUNC uint64_t libdivide_128_div_64_to_64(uint64_t numhi, uint64_t numlo, uint64_t den,
                                                   uint64_t* r) {
  // We work in base 2**32.
  // A uint32 holds a single digit. A uint64 holds two digits.
  // Our numerator is conceptually [num3, num2, num1, num0].
  // Our denominator is [den1, den0].
  const uint64_t b = ((uint64_t)1 << 32);

  // The high and low digits of our computed quotient.
  uint32_t q1;
  uint32_t q0;

  // The normalization shift factor.
  int shift;

  // The high and low digits of our denominator (after normalizing).
  // Also the low 2 digits of our numerator (after normalizing).
  uint32_t den1;
  uint32_t den0;
  uint32_t num1;
  uint32_t num0;

  // A partial remainder.
  uint64_t rem;

  // The estimated quotient, and its corresponding remainder (unrelated to true remainder).
  uint64_t qhat;
  uint64_t rhat;

  // Variables used to correct the estimated quotient.
  uint64_t c1;
  uint64_t c2;

  // Check for overflow and divide by 0.
  if (numhi >= den) {
    if (r != NULL) *r = ~0ull;
    return ~0ull;
  }

  // Determine the normalization factor. We multiply den by this, so that its leading digit is at
  // least half b. In binary this means just shifting left by the number of leading zeros, so that
  // there's a 1 in the MSB.
  // We also shift numer by the same amount. This cannot overflow because numhi < den.
  // The expression (-shift & 63) is the same as (64 - shift), except it avoids the UB of shifting
  // by 64. The funny bitwise 'and' ensures that numlo does not get shifted into numhi if shift is
  // 0. clang 11 has an x86 codegen bug here: see LLVM bug 50118. The sequence below avoids it.
  shift = libdivide_count_leading_zeros64(den);
  den <<= shift;
  numhi <<= shift;
  numhi |= (numlo >> (-shift & 63)) & (-(int64_t)shift >> 63);
  numlo <<= shift;

  // Extract the low digits of the numerator and both digits of the denominator.
  num1 = (uint32_t)(numlo >> 32);
  num0 = (uint32_t)(numlo & 0xFFFFFFFFu);
  den1 = (uint32_t)(den >> 32);
  den0 = (uint32_t)(den & 0xFFFFFFFFu);

  // We wish to compute q1 = [n3 n2 n1] / [d1 d0].
  // Estimate q1 as [n3 n2] / [d1], and then correct it.
  // Note while qhat may be 2 digits, q1 is always 1 digit.
  qhat = numhi / den1;
  rhat = numhi % den1;
  c1 = qhat * den0;
  c2 = rhat * b + num1;
  if (c1 > c2) qhat -= (c1 - c2 > den) ? 2 : 1;
  q1 = (uint32_t)qhat;

  // Compute the true (partial) remainder.
  rem = numhi * b + num1 - q1 * den;

  // We wish to compute q0 = [rem1 rem0 n0] / [d1 d0].
  // Estimate q0 as [rem1 rem0] / [d1] and correct it.
  qhat = rem / den1;
  rhat = rem % den1;
  c1 = qhat * den0;
  c2 = rhat * b + num0;
  if (c1 > c2) qhat -= (c1 - c2 > den) ? 2 : 1;
  q0 = (uint32_t)qhat;

  // Return remainder if requested.
  if (r != NULL) *r = (rem * b + num0 - q0 * den) >> shift;
  return ((uint64_t)q1 << 32) | q0;
}

OF_DEVICE_FUNC uint32_t libdivide_mullhi_u32(uint32_t x, uint32_t y) {
  uint64_t xl = x, yl = y;
  uint64_t rl = xl * yl;
  return (uint32_t)(rl >> 32);
}

OF_DEVICE_FUNC int64_t libdivide_mullhi_s64(int64_t x, int64_t y) {
  uint32_t mask = 0xFFFFFFFF;
  uint32_t x0 = (uint32_t)(x & mask);
  uint32_t y0 = (uint32_t)(y & mask);
  int32_t x1 = (int32_t)(x >> 32);
  int32_t y1 = (int32_t)(y >> 32);
  uint32_t x0y0_hi = libdivide_mullhi_u32(x0, y0);
  int64_t t = x1 * (int64_t)y0 + x0y0_hi;
  int64_t w1 = x0 * (int64_t)y1 + (t & mask);

  return x1 * (int64_t)y1 + (t >> 32) + (w1 >> 32);
}

}  // namespace

template<typename T>
class FastDivide {
 public:
  OF_DEVICE_FUNC FastDivide() = default;

  OF_DEVICE_FUNC FastDivide(T operand) { this->operand_ = operand; }

  OF_DEVICE_FUNC T divides(uint64_t x) const { return x / operand_; }

 private:
  T operand_;
};

template<>
class FastDivide<uint64_t> {
 public:
  OF_DEVICE_FUNC FastDivide() = default;

  OF_DEVICE_FUNC FastDivide(uint64_t operand) { init(operand); }

  OF_DEVICE_FUNC void init(uint64_t operand) {
    assert(operand != 0);
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

  OF_DEVICE_FUNC uint64_t divides(uint64_t x) const {
    // If the operand is a power of 2, the multiplier would be 2^64, which is out of range
    // - therefore, powers of 2 get special treatment and the multiplication is skipped.
    x = static_cast<uint64_t>(x);
#ifdef __CUDA_ARCH__
    if (mul_factor_) { x = __umul64hi(x + add_, mul_factor_); }
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

 private:
  uint64_t mul_factor_;
  uint8_t add_;
  uint8_t shift_;
};

template<>
class FastDivide<uint32_t> {
 public:
  OF_DEVICE_FUNC FastDivide() = default;

  OF_DEVICE_FUNC FastDivide(uint32_t operand) { init(operand); }

  OF_DEVICE_FUNC void init(uint32_t operand) {
    assert(operand != 0);
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

  OF_DEVICE_FUNC uint32_t divides(uint32_t x) const {
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

 private:
  uint32_t mul_factor_;
  uint8_t add_;
  uint8_t shift_;
};

template<>
class FastDivide<int32_t> {
 public:
  OF_DEVICE_FUNC FastDivide() = default;
  OF_DEVICE_FUNC explicit FastDivide(int32_t operand) {
    assert(operand != 0);
    int branchfree = 0;
    uint32_t ud = (uint32_t)operand;
    uint32_t absD = (operand < 0) ? -ud : ud;
#if defined(__CUDA_ARCH__)
    uint32_t floor_log_2_d = 31 - __clz(operand);
#else
    uint32_t floor_log_2_d = 31 - __builtin_clz(operand);
#endif

    // Power of 2
    if ((absD & (absD - 1)) == 0) {
      // Branchfree and normal paths are exactly the same
      magic_ = 0;
      more_ = (uint8_t)(floor_log_2_d | (operand < 0 ? LIBDIVIDE_NEGATIVE_DIVISOR : 0));
    } else {
      assert(floor_log_2_d >= 1);
      uint8_t more;
      // the dividend here is 2**(floor_log_2_d + 31), so the low 32 bit word
      // is 0 and the high word is floor_log_2_d - 1
      uint32_t rem, proposed_m;
      proposed_m = libdivide_64_div_32_to_32((uint32_t)1 << (floor_log_2_d - 1), 0, absD, &rem);
      const uint32_t e = absD - rem;
      if (!branchfree && e < ((uint32_t)1 << floor_log_2_d)) {
        more = (uint8_t)(floor_log_2_d - 1);
      } else {
        proposed_m += proposed_m;
        const uint32_t twice_rem = rem + rem;
        if (twice_rem >= absD || twice_rem < rem) proposed_m += 1;
        more = (uint8_t)(floor_log_2_d | LIBDIVIDE_ADD_MARKER);
      }

      proposed_m += 1;
      int32_t magic = (int32_t)proposed_m;
      if (operand < 0) {
        more |= LIBDIVIDE_NEGATIVE_DIVISOR;
        if (!branchfree) { magic = -magic; }
      }
      more_ = more;
      magic_ = magic;
    }
  }

  OF_DEVICE_FUNC int32_t divides(const int32_t numer) const {
    uint8_t more = more_;
    uint8_t shift = more & LIBDIVIDE_32_SHIFT_MASK;

    if (!magic_) {
      uint32_t sign = (int8_t)more >> 7;
      uint32_t mask = ((uint32_t)1 << shift) - 1;
      uint32_t uq = numer + ((numer >> 31) & mask);
      int32_t q = (int32_t)uq;
      q >>= shift;
      q = (q ^ sign) - sign;
      return q;
    } else {
      uint32_t uq = (uint32_t)libdivide_mullhi_s32(magic_, numer);
      if (more & LIBDIVIDE_ADD_MARKER) {
        int32_t sign = (int8_t)more >> 7;
        uq += ((uint32_t)numer ^ sign) - sign;
      }
      int32_t q = (int32_t)uq;
      q >>= shift;
      q += (q < 0);
      return q;
    }
  }

 private:
  int32_t magic_;
  uint8_t more_;
};

template<>
class FastDivide<int64_t> {
 public:
  OF_DEVICE_FUNC FastDivide() = default;
  OF_DEVICE_FUNC explicit FastDivide(int64_t operand) {
    int branchfree = 0;
    uint64_t ud = (uint64_t)operand;
    uint64_t absD = (operand < 0) ? -ud : ud;

#if defined(__CUDA_ARCH__)
    uint32_t floor_log_2_d = 63 - __clzll(absD);
#else
    uint32_t floor_log_2_d = 63 - __builtin_clzll(absD);
#endif
    // Power of 2
    if ((absD & (absD - 1)) == 0) {
      // Branchfree and non-branchfree cases are the same
      magic_ = 0;
      more_ = (uint8_t)(floor_log_2_d | (operand < 0 ? LIBDIVIDE_NEGATIVE_DIVISOR : 0));
    } else {
      // the dividend here is 2**(floor_log_2_d + 63), so the low 64 bit word
      // is 0 and the high word is floor_log_2_d - 1
      uint8_t more;
      uint64_t rem, proposed_m;
      proposed_m = libdivide_128_div_64_to_64((uint64_t)1 << (floor_log_2_d - 1), 0, absD, &rem);
      const uint64_t e = absD - rem;

      // We are going to start with a power of floor_log_2_d - 1.
      // This works if works if e < 2**floor_log_2_d.
      if (!branchfree && e < ((uint64_t)1 << floor_log_2_d)) {
        // This power works
        more = (uint8_t)(floor_log_2_d - 1);
      } else {
        // We need to go one higher. This should not make proposed_m
        // overflow, but it will make it negative when interpreted as an
        // int32_t.
        proposed_m += proposed_m;
        const uint64_t twice_rem = rem + rem;
        if (twice_rem >= absD || twice_rem < rem) proposed_m += 1;
        // note that we only set the LIBDIVIDE_NEGATIVE_DIVISOR bit if we
        // also set ADD_MARKER this is an annoying optimization that
        // enables algorithm #4 to avoid the mask. However we always set it
        // in the branchfree case
        more = (uint8_t)(floor_log_2_d | LIBDIVIDE_ADD_MARKER);
      }
      proposed_m += 1;
      int64_t magic = (int64_t)proposed_m;

      // Mark if we are negative
      if (operand < 0) {
        more |= LIBDIVIDE_NEGATIVE_DIVISOR;
        if (!branchfree) { magic = -magic; }
      }
      more_ = more;
      magic_ = magic;
    }
  }

  OF_DEVICE_FUNC int64_t divides(const int64_t numer) const {
    uint8_t more = more_;
    uint8_t shift = more & LIBDIVIDE_64_SHIFT_MASK;

    if (!magic_) {  // shift path
      uint64_t mask = ((uint64_t)1 << shift) - 1;
      uint64_t uq = numer + ((numer >> 63) & mask);
      int64_t q = (int64_t)uq;
      q >>= shift;
      // must be arithmetic shift and then sign-extend
      int64_t sign = (int8_t)more >> 7;
      q = (q ^ sign) - sign;
      return q;
    } else {
      uint64_t uq = (uint64_t)libdivide_mullhi_s64(magic_, numer);
      if (more & LIBDIVIDE_ADD_MARKER) {
        // must be arithmetic shift and then sign extend
        int64_t sign = (int8_t)more >> 7;
        // q += (more < 0 ? -numer : numer)
        // cast required to avoid UB
        uq += ((uint64_t)numer ^ sign) - sign;
      }
      int64_t q = (int64_t)uq;
      q >>= shift;
      q += (q < 0);
      return q;
    }
  }

 private:
  int64_t magic_;
  uint8_t more_;
};

template<typename T, int N>
void InitStrides(const int64_t* dims, T* strides, int n) {
  for (int i = n - 1; i < N; ++i) { strides[i] = 1; }
  for (int i = n - 2; i >= 0; --i) { strides[i] = dims[i + 1] * strides[i + 1]; }
}

template<typename T>
void InitFastDividers(T* strides, FastDivide<T>* fast_dividers, int n) {
  for (int i = n - 1; i >= 0; --i) { fast_dividers[i] = FastDivide<T>(strides[i]); }
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_EP_PRIMITIVE_FAST_INTEGER_MATH_H_
