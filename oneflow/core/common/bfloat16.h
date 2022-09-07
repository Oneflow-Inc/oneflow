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
#ifndef ONEFLOW_CORE_COMMON_BFLOAT16_H_
#define ONEFLOW_CORE_COMMON_BFLOAT16_H_

#include <stdint.h>
#include <limits>
#include <cmath>
#include <cstring>
#if defined(WITH_CUDA)
#if CUDA_VERSION >= 11000
#include <cuda_bf16.h>
#endif  // CUDA_VERSION >= 11000
#endif

namespace oneflow {

#if defined(__CUDACC__)
#define OF_DEVICE_FUNCTION __device__ __host__ __forceinline__
#else
#define OF_DEVICE_FUNCTION inline
#endif

struct alignas(2) bfloat16 {
  uint16_t x;

  bfloat16() = default;
  bfloat16(const bfloat16& o) = default;
  bfloat16& operator=(const bfloat16& o) = default;
  bfloat16(bfloat16&& o) = default;
  bfloat16& operator=(bfloat16&& o) = default;
  ~bfloat16() = default;

  struct from_bits_t {};
  static constexpr OF_DEVICE_FUNCTION from_bits_t from_bits() { return from_bits_t(); }

  constexpr OF_DEVICE_FUNCTION bfloat16(unsigned short bits, from_bits_t) : x(bits){};

  // reference: pytorch/c10/util/BFloat16.h
  // https://github.com/pytorch/pytorch/blob/release/1.12/c10/util/BFloat16.h
  OF_DEVICE_FUNCTION bfloat16(float value) {
    if (std::isnan(value)) {
      x = 0x7FC0;
    } else {
      union {
        uint32_t U32;
        float F32;
      };

      F32 = value;
      uint32_t rounding_bias = ((U32 >> 16) & 1) + 0x7FFFU;
      x = static_cast<uint16_t>((U32 + rounding_bias) >> 16);
    }
  }

#if defined(WITH_CUDA)
#if CUDA_VERSION >= 11000
  OF_DEVICE_FUNCTION bfloat16(const __nv_bfloat16& value) {
    x = *reinterpret_cast<const unsigned short*>(&value);
  }
  explicit OF_DEVICE_FUNCTION operator __nv_bfloat16() const {
    return *reinterpret_cast<const __nv_bfloat16*>(&x);
  }
#endif  // CUDA_VERSION >= 11000
#endif  // WITH_CUDA

  OF_DEVICE_FUNCTION operator float() const {
#if defined(WITH_CUDA)
#if CUDA_VERSION >= 11000
    return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&x));
#endif  // CUDA_VERSION >= 11000
#endif  // WITH_CUDA
    float res = 0;
    uint32_t tmp = x;
    tmp <<= 16;
    std::memcpy(&res, &tmp, sizeof(tmp));
    return res;
  }

  inline bool operator==(const bfloat16& other) const { return x == other.x; }

  OF_DEVICE_FUNCTION explicit operator bool() const { return (x & 0x7fff) != 0; }

  OF_DEVICE_FUNCTION explicit operator int8_t() const {
    return static_cast<int8_t>(static_cast<float>(*this));
  }

  OF_DEVICE_FUNCTION explicit operator uint8_t() const {
    return static_cast<uint8_t>(static_cast<float>(*this));
  }

  OF_DEVICE_FUNCTION explicit operator int16_t() const {
    return static_cast<int16_t>(static_cast<float>(*this));
  }

  OF_DEVICE_FUNCTION explicit operator uint16_t() const {
    return static_cast<uint16_t>(static_cast<float>(*this));
  }

  OF_DEVICE_FUNCTION explicit operator int32_t() const {
    return static_cast<int32_t>(static_cast<float>(*this));
  }

  OF_DEVICE_FUNCTION explicit operator uint32_t() const {
    return static_cast<uint32_t>(static_cast<float>(*this));
  }

  OF_DEVICE_FUNCTION explicit operator int64_t() const {
    return static_cast<int64_t>(static_cast<float>(*this));
  }

  OF_DEVICE_FUNCTION explicit operator uint64_t() const {
    return static_cast<uint64_t>(static_cast<float>(*this));
  }

  OF_DEVICE_FUNCTION explicit operator double() const {
    return static_cast<double>(static_cast<float>(*this));
  }
};

// Arithmetic

OF_DEVICE_FUNCTION bfloat16 operator+(const bfloat16& a, const bfloat16& b) {
  return static_cast<float>(a) + static_cast<float>(b);
}

OF_DEVICE_FUNCTION bfloat16 operator-(const bfloat16& a, const bfloat16& b) {
  return static_cast<float>(a) - static_cast<float>(b);
}

OF_DEVICE_FUNCTION bfloat16 operator*(const bfloat16& a, const bfloat16& b) {
  return static_cast<float>(a) * static_cast<float>(b);
}

OF_DEVICE_FUNCTION bfloat16 operator/(const bfloat16& a, const bfloat16& b) {
  return static_cast<float>(a) / static_cast<float>(b);
}

OF_DEVICE_FUNCTION bfloat16 operator-(const bfloat16& a) {
  bfloat16 output;
  output.x = a.x ^ 0x8000U;
  return output;
}

OF_DEVICE_FUNCTION bfloat16& operator+=(bfloat16& a, const bfloat16& b) {
  a = a + b;
  return a;
}

OF_DEVICE_FUNCTION bfloat16& operator-=(bfloat16& a, const bfloat16& b) {
  a = a - b;
  return a;
}

OF_DEVICE_FUNCTION bfloat16& operator*=(bfloat16& a, const bfloat16& b) {
  a = a * b;
  return a;
}

OF_DEVICE_FUNCTION bfloat16& operator/=(bfloat16& a, const bfloat16& b) {
  a = a / b;
  return a;
}

OF_DEVICE_FUNCTION bfloat16& operator|(bfloat16& a, const bfloat16& b) {
  a.x = a.x | b.x;
  return a;
}

OF_DEVICE_FUNCTION bfloat16& operator^(bfloat16& a, const bfloat16& b) {
  a.x = a.x ^ b.x;
  return a;
}

OF_DEVICE_FUNCTION bfloat16& operator&(bfloat16& a, const bfloat16& b) {
  a.x = a.x & b.x;
  return a;
}

// Arithmetic with floats

OF_DEVICE_FUNCTION float operator+(bfloat16 a, float b) { return static_cast<float>(a) + b; }
OF_DEVICE_FUNCTION float operator-(bfloat16 a, float b) { return static_cast<float>(a) - b; }
OF_DEVICE_FUNCTION float operator*(bfloat16 a, float b) { return static_cast<float>(a) * b; }
OF_DEVICE_FUNCTION float operator/(bfloat16 a, float b) { return static_cast<float>(a) / b; }

OF_DEVICE_FUNCTION float operator+(float a, bfloat16 b) { return a + static_cast<float>(b); }
OF_DEVICE_FUNCTION float operator-(float a, bfloat16 b) { return a - static_cast<float>(b); }
OF_DEVICE_FUNCTION float operator*(float a, bfloat16 b) { return a * static_cast<float>(b); }
OF_DEVICE_FUNCTION float operator/(float a, bfloat16 b) { return a / static_cast<float>(b); }

OF_DEVICE_FUNCTION float& operator+=(float& a, const bfloat16& b) {
  return a += static_cast<float>(b);
}
OF_DEVICE_FUNCTION float& operator-=(float& a, const bfloat16& b) {
  return a -= static_cast<float>(b);
}
OF_DEVICE_FUNCTION float& operator*=(float& a, const bfloat16& b) {
  return a *= static_cast<float>(b);
}
OF_DEVICE_FUNCTION float& operator/=(float& a, const bfloat16& b) {
  return a /= static_cast<float>(b);
}

// Arithmetic with doubles

OF_DEVICE_FUNCTION double operator+(bfloat16 a, double b) { return static_cast<double>(a) + b; }
OF_DEVICE_FUNCTION double operator-(bfloat16 a, double b) { return static_cast<double>(a) - b; }
OF_DEVICE_FUNCTION double operator*(bfloat16 a, double b) { return static_cast<double>(a) * b; }
OF_DEVICE_FUNCTION double operator/(bfloat16 a, double b) { return static_cast<double>(a) / b; }

OF_DEVICE_FUNCTION double operator+(double a, bfloat16 b) { return a + static_cast<double>(b); }
OF_DEVICE_FUNCTION double operator-(double a, bfloat16 b) { return a - static_cast<double>(b); }
OF_DEVICE_FUNCTION double operator*(double a, bfloat16 b) { return a * static_cast<double>(b); }
OF_DEVICE_FUNCTION double operator/(double a, bfloat16 b) { return a / static_cast<double>(b); }

// Arithmetic with int32_t

OF_DEVICE_FUNCTION bfloat16 operator+(bfloat16 a, int32_t b) {
  return a + static_cast<bfloat16>(b);
}
OF_DEVICE_FUNCTION bfloat16 operator-(bfloat16 a, int32_t b) {
  return a - static_cast<bfloat16>(b);
}
OF_DEVICE_FUNCTION bfloat16 operator*(bfloat16 a, int32_t b) {
  return a * static_cast<bfloat16>(b);
}
OF_DEVICE_FUNCTION bfloat16 operator/(bfloat16 a, int32_t b) {
  return a / static_cast<bfloat16>(b);
}

OF_DEVICE_FUNCTION bfloat16 operator+(int32_t a, bfloat16 b) {
  return static_cast<bfloat16>(a) + b;
}
OF_DEVICE_FUNCTION bfloat16 operator-(int32_t a, bfloat16 b) {
  return static_cast<bfloat16>(a) - b;
}
OF_DEVICE_FUNCTION bfloat16 operator*(int32_t a, bfloat16 b) {
  return static_cast<bfloat16>(a) * b;
}
OF_DEVICE_FUNCTION bfloat16 operator/(int32_t a, bfloat16 b) {
  return static_cast<bfloat16>(a) / b;
}

// Arithmetic with int64_t

OF_DEVICE_FUNCTION bfloat16 operator+(bfloat16 a, int64_t b) {
  return a + static_cast<bfloat16>(b);
}
OF_DEVICE_FUNCTION bfloat16 operator-(bfloat16 a, int64_t b) {
  return a - static_cast<bfloat16>(b);
}
OF_DEVICE_FUNCTION bfloat16 operator*(bfloat16 a, int64_t b) {
  return a * static_cast<bfloat16>(b);
}
OF_DEVICE_FUNCTION bfloat16 operator/(bfloat16 a, int64_t b) {
  return a / static_cast<bfloat16>(b);
}

OF_DEVICE_FUNCTION bfloat16 operator+(int64_t a, bfloat16 b) {
  return static_cast<bfloat16>(a) + b;
}
OF_DEVICE_FUNCTION bfloat16 operator-(int64_t a, bfloat16 b) {
  return static_cast<bfloat16>(a) - b;
}
OF_DEVICE_FUNCTION bfloat16 operator*(int64_t a, bfloat16 b) {
  return static_cast<bfloat16>(a) * b;
}
OF_DEVICE_FUNCTION bfloat16 operator/(int64_t a, bfloat16 b) {
  return static_cast<bfloat16>(a) / b;
}

// Comparison operators

OF_DEVICE_FUNCTION bool operator>(bfloat16& lhs, bfloat16& rhs) {
  return static_cast<float>(lhs) > static_cast<float>(rhs);
}

OF_DEVICE_FUNCTION bool operator>=(bfloat16& lhs, bfloat16& rhs) {
  return static_cast<float>(lhs) >= static_cast<float>(rhs);
}

OF_DEVICE_FUNCTION bool operator<(bfloat16& lhs, bfloat16& rhs) {
  return static_cast<float>(lhs) < static_cast<float>(rhs);
}

OF_DEVICE_FUNCTION bool operator<=(bfloat16& lhs, bfloat16& rhs) {
  return static_cast<float>(lhs) <= static_cast<float>(rhs);
}

OF_DEVICE_FUNCTION bool operator==(bfloat16& lhs, bfloat16& rhs) {
  return static_cast<float>(lhs) == static_cast<float>(rhs);
}

OF_DEVICE_FUNCTION bool operator!=(bfloat16& lhs, bfloat16& rhs) {
  return static_cast<float>(lhs) != static_cast<float>(rhs);
}

}  // namespace oneflow

namespace std {

inline bool isnan(const oneflow::bfloat16& value) { return (value.x & 0x7FFFU) > 0x07F80U; }

inline bool isinf(const oneflow::bfloat16& value) { return value.x == 0x07F80U; }

inline bool isfinite(const oneflow::bfloat16& value) { return !isinf(value) && !isnan(value); }

template<>
class numeric_limits<oneflow::bfloat16> {
 public:
  static constexpr bool is_signed = true;
  static constexpr bool is_specialized = true;
  static constexpr bool is_integer = false;
  static constexpr bool is_exact = false;
  static constexpr bool has_infinity = true;
  static constexpr bool has_quiet_NaN = true;
  static constexpr bool has_signaling_NaN = true;
  static constexpr auto has_denorm = numeric_limits<float>::has_denorm;
  static constexpr auto has_denorm_loss = numeric_limits<float>::has_denorm_loss;
  static constexpr auto round_style = numeric_limits<float>::round_style;
  static constexpr bool is_iec559 = false;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo = false;
  static constexpr int digits = 8;
  static constexpr int digits10 = 2;
  static constexpr int max_digits10 = 4;
  static constexpr int radix = 2;
  static constexpr int min_exponent = -125;
  static constexpr int min_exponent10 = -37;
  static constexpr int max_exponent = 128;
  static constexpr int max_exponent10 = 38;
  static constexpr auto traps = numeric_limits<float>::traps;
  static constexpr auto tinyness_before = numeric_limits<float>::tinyness_before;
  static constexpr oneflow::bfloat16 min() {
    return oneflow::bfloat16(0x0080U, oneflow::bfloat16::from_bits());
  }
  static constexpr oneflow::bfloat16 lowest() {
    return oneflow::bfloat16(0xFF7FU, oneflow::bfloat16::from_bits());
  }
  static constexpr oneflow::bfloat16 max() {
    return oneflow::bfloat16(0x7F7FU, oneflow::bfloat16::from_bits());
  }
  static constexpr oneflow::bfloat16 epsilon() {
    return oneflow::bfloat16(0x3C00U, oneflow::bfloat16::from_bits());
  }
  static constexpr oneflow::bfloat16 round_error() {
    return oneflow::bfloat16(0x3F00U, oneflow::bfloat16::from_bits());
  }
  static constexpr oneflow::bfloat16 infinity() {
    return oneflow::bfloat16(0x7F80U, oneflow::bfloat16::from_bits());
  }
  static constexpr oneflow::bfloat16 quiet_NaN() {
    return oneflow::bfloat16(0x7FC0U, oneflow::bfloat16::from_bits());
  }
  static constexpr oneflow::bfloat16 signaling_NaN() {
    return oneflow::bfloat16(0x7F80U, oneflow::bfloat16::from_bits());
  }
  static constexpr oneflow::bfloat16 denorm_min() {
    return oneflow::bfloat16(0x0001U, oneflow::bfloat16::from_bits());
  }
};

}  // namespace std

#endif  // ONEFLOW_CORE_COMMON_BFLOAT16_H_
