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
#include "oneflow/core/ep/common/primitive/unary_functor.h"
#include "oneflow/core/ep/cpu/primitive/type_seq.h"
#include "oneflow/core/common/math_util.h"

namespace oneflow {
namespace ep {
namespace primitive {

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCPU, UnaryOp::kGelu, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const {
    return static_cast<Src>(0.5) * src * (static_cast<Src>(1.0) + std::erf(inv_sqrt2 * src));
  }
  Src inv_sqrt2 = std::sqrt(0.5);
};

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCPU, UnaryOp::kFastGelu, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const {
    // ref to: https://mlfromscratch.com/activation-functions-explained/#gelu
    const Src half = static_cast<Src>(0.5);
    const Src one = static_cast<Src>(1);
    const Src tanh_in = alpha * (src + beta * src * src * src);
    return half * src * (one + std::tanh(tanh_in));
  }

 private:
  // constant ref to:
  // https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/test/testdata/transform/fusion/fast_gelu.py
  static constexpr Src alpha = static_cast<Src>(0.7978845608028654);
  static constexpr Src beta = static_cast<Src>(0.044714998453855515);
};

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCPU, UnaryOp::kQuickGelu, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const {
    const Src sigmoid =
        static_cast<Dst>(static_cast<Src>(1.0) / (static_cast<Src>(1.0) + exp(-src * alpha)));
    return src * sigmoid;
  }

 private:
  static constexpr Src alpha = static_cast<Src>(1.702);
};

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCPU, UnaryOp::kTanh, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const { return std::tanh(src); }
};

template<>
struct UnaryFunctor<DeviceType::kCPU, UnaryOp::kIsInf, bool, float> {
  UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC bool operator()(float src) const { return std::isinf(src); }
};

template<>
struct UnaryFunctor<DeviceType::kCPU, UnaryOp::kIsInf, bool, double> {
  UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC bool operator()(double src) const { return std::isinf(src); }
};

template<>
struct UnaryFunctor<DeviceType::kCPU, UnaryOp::kIsNan, bool, float> {
  UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC bool operator()(float src) const { return std::isnan(src); }
};

template<>
struct UnaryFunctor<DeviceType::kCPU, UnaryOp::kIsNan, bool, double> {
  UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC bool operator()(double src) const { return std::isnan(src); }
};

template<typename Src>
struct UnaryFunctor<DeviceType::kCPU, UnaryOp::kIsFinite, bool, Src> {
  UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC bool operator()(Src src) const { return std::isfinite(src); }
};

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCPU, UnaryOp::kTrunc, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}
  OF_DEVICE_FUNC Dst operator()(Src src) const { return static_cast<Dst>(std::trunc(src)); }
};

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCPU, UnaryOp::kRsqrt, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const {
    return static_cast<Dst>(static_cast<Src>(1.0) / static_cast<Src>(std::sqrt(src)));
  }
};

template<>
struct UnaryFunctor<DeviceType::kCPU, UnaryOp::kDigamma, float, float> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC float operator()(float src) const {
    // references
    // https://github.com/pytorch/pytorch/blob/release/1.13/aten/src/ATen/native/Math.h#L434-L487
    const auto& calc_digamma = [](float x) {
      std::function<float(float)> compute;
      compute = [&](float x) {
        static float PSI_10 = 2.25175258906672110764f;
        if (x == 0) {
          // As per C++ standard for gamma related functions and SciPy,
          // If the argument is ±0, ±∞ is returned
          return std::copysign(INFINITY, -x);
        }

        bool x_is_integer = x == truncf(x);
        if (x < 0) {
          if (x_is_integer) {
            // As per C++ standard for gamma related functions and SciPy,
            // If the argument is a negative integer, NaN is returned
            return std::numeric_limits<float>::quiet_NaN();
          }
          // Extracts the fractional part of x as r, since tan(pi * r) is more numerically
          // accurate than tan(pi * x). While these operations are mathematically equivalent
          // since both x and r are in radians and tan() has a periodicity of pi, in practice
          // the computation of pi * x is a source of error (when |x| > 1).
          double q, r;
          r = std::modf(x, &q);
          float pi_over_tan_pi_x = (float)(pi<double> / tan(pi<double> * r));
          return compute(1 - x) - pi_over_tan_pi_x;
        }

        // Push x to be >= 10
        float result = 0;
        while (x < 10) {
          result -= 1 / x;
          x += 1;
        }
        if (x == 10) { return result + PSI_10; }

        // Compute asymptotic digamma
        static const float A[] = {
            8.33333333333333333333E-2f,  -2.10927960927960927961E-2f, 7.57575757575757575758E-3f,
            -4.16666666666666666667E-3f, 3.96825396825396825397E-3f,  -8.33333333333333333333E-3f,
            8.33333333333333333333E-2f,
        };

        float y = 0;
        if (x < 1.0e17f) {
          float z = 1 / (x * x);
          float polevl_result = 0;
          for (int i = 0; i <= 6; i++) { polevl_result = polevl_result * z + A[i]; }
          y = z * polevl_result;
        }
        return result + logf(x) - (0.5f / x) - y;
      };

      return compute(x);
    };

    return calc_digamma(src);
  }
};

template<>
struct UnaryFunctor<DeviceType::kCPU, UnaryOp::kDigamma, double, double> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC double operator()(double src) const {
    // references
    // https://github.com/pytorch/pytorch/blob/release/1.13/aten/src/ATen/native/Math.h#L376-L428
    const auto& calc_digamma = [](double x) {
      std::function<double(double)> compute;
      compute = [&](double x) {
        static double PSI_10 = 2.25175258906672110764;
        if (x == 0) {
          // As per C++ standard for gamma related functions and SciPy,
          // If the argument is ±0, ±∞ is returned
          return std::copysign(INFINITY, -x);
        }

        bool x_is_integer = x == trunc(x);
        if (x < 0) {
          if (x_is_integer) {
            // As per C++ standard for gamma related functions and SciPy,
            // If the argument is a negative integer, NaN is returned
            return std::numeric_limits<double>::quiet_NaN();
          }
          // Extracts the fractional part of x as r, since tan(pi * r) is more numerically
          // accurate than tan(pi * x). While these operations are mathematically equivalent
          // since both x and r are in radians and tan() has a periodicity of pi, in practice
          // the computation of pi * x is a source of error (when |x| > 1).
          double q, r;
          r = std::modf(x, &q);
          return compute(1 - x) - pi<double> / tan(pi<double> * r);
        }

        // Push x to be >= 10
        double result = 0;
        while (x < 10) {
          result -= 1 / x;
          x += 1;
        }
        if (x == 10) { return result + PSI_10; }

        // Compute asymptotic digamma
        static const double A[] = {
            8.33333333333333333333E-2,  -2.10927960927960927961E-2, 7.57575757575757575758E-3,
            -4.16666666666666666667E-3, 3.96825396825396825397E-3,  -8.33333333333333333333E-3,
            8.33333333333333333333E-2,
        };

        double y = 0;
        if (x < 1.0e17) {
          double z = 1.0 / (x * x);
          // y = z * polevl(z, A, 6);

          double polevl_result = 0;
          for (int i = 0; i <= 6; i++) { polevl_result = polevl_result * z + A[i]; }
          y = z * polevl_result;
        }
        return result + log(x) - (0.5 / x) - y;
      };

      return compute(x);
    };

    return calc_digamma(src);
  }
};

template<>
struct UnaryFunctor<DeviceType::kCPU, UnaryOp::kTrigamma, double, double> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC double operator()(double x) const {
    // references
    // https://github.com/pytorch/pytorch/blob/release/1.13/aten/src/ATen/native/Math.h#L336-L352
    double sign = +1;
    double result = 0;
    if (x < 0.5) {
      sign = -1;
      const double sin_pi_x = sin(pi<double> * x);
      result -= (pi<double> * pi<double>) / (sin_pi_x * sin_pi_x);
      x = 1 - x;
    }
    for (int i = 0; i < 6; ++i) {
      result += 1 / (x * x);
      x += 1;
    }
    const double ixx = 1 / (x * x);
    result += (1 + 1 / (2 * x) + ixx * (1. / 6 - ixx * (1. / 30 - ixx * (1. / 42)))) / x;
    return sign * result;
  }
};

template<>
struct UnaryFunctor<DeviceType::kCPU, UnaryOp::kTrigamma, float, float> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC float operator()(float x) const {
    // references
    // https://github.com/pytorch/pytorch/blob/release/1.13/aten/src/ATen/native/Math.h#L354-L370
    float sign = +1;
    float result = 0;
    if (x < 0.5f) {
      sign = -1;
      const float sin_pi_x = sinf(pi<float> * x);
      result -= (pi<float> * pi<float>) / (sin_pi_x * sin_pi_x);
      x = 1 - x;
    }
    for (int i = 0; i < 6; ++i) {
      result += 1 / (x * x);
      x += 1;
    }
    const float ixx = 1 / (x * x);
    result += (1 + 1 / (2 * x) + ixx * (1.f / 6 - ixx * (1.f / 30 - ixx * (1.f / 42)))) / x;
    return sign * result;
  }
};

template<>
struct UnaryFunctor<DeviceType::kCPU, UnaryOp::kAbs, bfloat16, bfloat16> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC bfloat16 operator()(bfloat16 src) const { return std::abs(src); }
};

#define SPECIALIZATION_CPU_BFLOAT16_UNARY_FUNCTOR(op)                                        \
  template<>                                                                                 \
  struct UnaryFunctor<DeviceType::kCPU, op, bfloat16, bfloat16> {                            \
    OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) : float_functor(attr0, attr1) {} \
                                                                                             \
    UnaryFunctor<DeviceType::kCPU, op, float, float> float_functor;                          \
    OF_DEVICE_FUNC bfloat16 operator()(bfloat16 src) const {                                 \
      return bfloat16(float_functor(static_cast<float>(src)));                               \
    }                                                                                        \
  };

SPECIALIZATION_CPU_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kElu);
SPECIALIZATION_CPU_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kCelu);
SPECIALIZATION_CPU_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kGelu);
SPECIALIZATION_CPU_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kHardSwish);
SPECIALIZATION_CPU_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kHardSigmoid);
SPECIALIZATION_CPU_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kHardShrink);
SPECIALIZATION_CPU_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kHardTanh);
SPECIALIZATION_CPU_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kLeakyRelu);
SPECIALIZATION_CPU_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kMish);
SPECIALIZATION_CPU_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kSelu);
SPECIALIZATION_CPU_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kSilu);
SPECIALIZATION_CPU_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kSoftShrink);
SPECIALIZATION_CPU_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kSoftSign);
SPECIALIZATION_CPU_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kSoftPlus);
SPECIALIZATION_CPU_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kTanh);
SPECIALIZATION_CPU_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kThreshold);
SPECIALIZATION_CPU_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kAcos);
SPECIALIZATION_CPU_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kAcosh);
SPECIALIZATION_CPU_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kAsin);
SPECIALIZATION_CPU_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kAsinh);
SPECIALIZATION_CPU_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kAtan);
SPECIALIZATION_CPU_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kAtanh);
SPECIALIZATION_CPU_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kCeil);
SPECIALIZATION_CPU_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kCos);
SPECIALIZATION_CPU_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kCosh);
SPECIALIZATION_CPU_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kErf);
SPECIALIZATION_CPU_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kErfc);
SPECIALIZATION_CPU_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kExp);
SPECIALIZATION_CPU_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kExp2);
SPECIALIZATION_CPU_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kExpm1);
SPECIALIZATION_CPU_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kFloor);
SPECIALIZATION_CPU_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kLgamma);
SPECIALIZATION_CPU_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kLog);
SPECIALIZATION_CPU_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kLog2);
SPECIALIZATION_CPU_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kLog1p);
SPECIALIZATION_CPU_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kLogSigmoid);
SPECIALIZATION_CPU_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kRint);
SPECIALIZATION_CPU_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kRound);
SPECIALIZATION_CPU_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kRsqrt);
SPECIALIZATION_CPU_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kSigmoid);
SPECIALIZATION_CPU_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kSin);
SPECIALIZATION_CPU_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kSinh);
SPECIALIZATION_CPU_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kSqrt);
SPECIALIZATION_CPU_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kSquare);
SPECIALIZATION_CPU_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kTan);
SPECIALIZATION_CPU_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kReciprocalNoNan);
SPECIALIZATION_CPU_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kNotEqualZero);
SPECIALIZATION_CPU_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kFastGelu);
SPECIALIZATION_CPU_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kQuickGelu);
SPECIALIZATION_CPU_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kDigamma);
SPECIALIZATION_CPU_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kTrigamma);

template<>
struct UnaryFunctor<DeviceType::kCPU, UnaryOp::kIsInf, bool, bfloat16> {
  UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC bool operator()(bfloat16 src) const { return std::isinf(src); }
};

template<>
struct UnaryFunctor<DeviceType::kCPU, UnaryOp::kIsNan, bool, bfloat16> {
  UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC bool operator()(bfloat16 src) const { return std::isnan(src); }
};

// avoid warning: narrowing conversion
template<>
struct UnaryFunctor<DeviceType::kCPU, UnaryOp::kRealGrad, std::complex<float>, double> {
  UnaryFunctor(Scalar attr0, Scalar attr1) {}
  std::complex<float> operator()(double src) const {
    return std::complex<float>{static_cast<float>(src), 0.0f};
  }
};

template<>
struct UnaryFunctor<DeviceType::kCPU, UnaryOp::kImagGrad, std::complex<float>, double> {
  UnaryFunctor(Scalar attr0, Scalar attr1) {}
  std::complex<float> operator()(double src) const {
    return std::complex<float>{0.0f, static_cast<float>(src)};
  }
};

}  // namespace primitive
}  // namespace ep
}  // namespace oneflow
