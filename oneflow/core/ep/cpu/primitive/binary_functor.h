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
#include "oneflow/core/ep/common/primitive/binary_functor.h"
#include "oneflow/core/ep/cpu/primitive/unary_functor.h"
namespace oneflow {

namespace ep {
namespace primitive {
namespace broadcast_elementwise_binary {

template<typename Src, typename Dst>
struct BinaryFunctor<DeviceType::kCPU, BinaryOp::kPow, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src0, Src src1) const { return std::pow(src0, src1); }
};

template<>
struct BinaryFunctor<DeviceType::kCPU, BinaryOp::kPow, float16, float16> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC float16 operator()(float16 src0, float16 src1) const {
    return static_cast<float16>(std::pow(static_cast<float>(src0), static_cast<float>(src1)));
  }
};

template<>
struct BinaryFunctor<DeviceType::kCPU, BinaryOp::kFmod, float, float> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC float operator()(float src0, float src1) const { return std::fmod(src0, src1); }
};

template<>
struct BinaryFunctor<DeviceType::kCPU, BinaryOp::kFmod, double, double> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC double operator()(double src0, double src1) const { return std::fmod(src0, src1); }
};

template<>
struct BinaryFunctor<DeviceType::kCPU, BinaryOp::kFmod, float16, float16> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC float16 operator()(float16 src0, float16 src1) const {
    return static_cast<float16>(std::fmod(static_cast<float>(src0), static_cast<float>(src1)));
  }
};

template<>
struct BinaryFunctor<DeviceType::kCPU, BinaryOp::kFmod, bfloat16, bfloat16> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC bfloat16 operator()(bfloat16 src0, bfloat16 src1) const {
    return std::fmod(src0, src1);
  }
};

template<>
struct BinaryFunctor<DeviceType::kCPU, BinaryOp::kFloorDiv, float, float> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC float operator()(float src0, float src1) const { return std::floor(src0 / src1); }
};

template<>
struct BinaryFunctor<DeviceType::kCPU, BinaryOp::kFloorDiv, double, double> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC double operator()(double src0, double src1) const {
    return std::floor(src0 / src1);
  }
};

template<>
struct BinaryFunctor<DeviceType::kCPU, BinaryOp::kFloorDiv, float16, float16> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC float16 operator()(float16 src0, float16 src1) const {
    return static_cast<float16>(std::floor(static_cast<float>(src0) / static_cast<float>(src1)));
  }
};

template<>
struct BinaryFunctor<DeviceType::kCPU, BinaryOp::kFloorDiv, bfloat16, bfloat16> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC bfloat16 operator()(bfloat16 src0, bfloat16 src1) const {
    return std::floor(src0 / src1);
  }
};

template<>
struct BinaryFunctor<DeviceType::kCPU, BinaryOp::kTruncDiv, float, float> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC float operator()(float src0, float src1) const { return std::trunc(src0 / src1); }
};

template<>
struct BinaryFunctor<DeviceType::kCPU, BinaryOp::kTruncDiv, double, double> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC double operator()(double src0, double src1) const {
    return std::trunc(src0 / src1);
  }
};

template<>
struct BinaryFunctor<DeviceType::kCPU, BinaryOp::kTruncDiv, float16, float16> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC float16 operator()(float16 src0, float16 src1) const {
    return static_cast<float16>(std::trunc(static_cast<float>(src0) / static_cast<float>(src1)));
  }
};

template<>
struct BinaryFunctor<DeviceType::kCPU, BinaryOp::kTruncDiv, bfloat16, bfloat16> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC bfloat16 operator()(bfloat16 src0, bfloat16 src1) const {
    return std::trunc(src0 / src1);
  }
};

template<>
struct BinaryFunctor<DeviceType::kCPU, BinaryOp::kFloorMod, float, float> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC float operator()(float src0, float src1) const {
    float trunc_mod = std::fmod(src0, src1);
    return (trunc_mod != static_cast<float>(0))
                   && ((src1 < static_cast<float>(0)) != (trunc_mod < static_cast<float>(0)))
               ? trunc_mod + src1
               : trunc_mod;
  }
};

template<>
struct BinaryFunctor<DeviceType::kCPU, BinaryOp::kFloorMod, double, double> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC double operator()(double src0, double src1) const {
    double trunc_mod = std::fmod(src0, src1);
    return (trunc_mod != static_cast<double>(0))
                   && ((src1 < static_cast<double>(0)) != (trunc_mod < static_cast<double>(0)))
               ? trunc_mod + src1
               : trunc_mod;
  }
};

template<>
struct BinaryFunctor<DeviceType::kCPU, BinaryOp::kFloorMod, float16, float16> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) : float_functor(attr0, attr1) {}
  BinaryFunctor<DeviceType::kCPU, BinaryOp::kFloorMod, float, float> float_functor;

  OF_DEVICE_FUNC float16 operator()(float16 src0, float16 src1) const {
    return static_cast<float16>(float_functor(static_cast<float>(src0), static_cast<float>(src1)));
  }
};

template<>
struct BinaryFunctor<DeviceType::kCPU, BinaryOp::kFloorMod, bfloat16, bfloat16> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) : float_functor(attr0, attr1) {}
  BinaryFunctor<DeviceType::kCPU, BinaryOp::kFloorMod, float, float> float_functor;

  OF_DEVICE_FUNC bfloat16 operator()(bfloat16 src0, bfloat16 src1) const {
    return static_cast<bfloat16>(float_functor(static_cast<float>(src0), static_cast<float>(src1)));
  }
};

template<>
struct BinaryFunctor<DeviceType::kCPU, BinaryOp::kScalarBasePowerGrad, float16, float16> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) : scalar_operand(attr0.Value<float>()) {}

  OF_DEVICE_FUNC float16 operator()(float16 src0, float16 src1) const {
    return static_cast<float16>(
        scalar_operand
        * (std::pow(static_cast<float>(src0), scalar_operand - static_cast<float>(1)))
        * static_cast<float>(src1));
  }
  float scalar_operand;
};

template<typename Dst>
struct BinaryFunctor<DeviceType::kCPU, BinaryOp::kScalarExpPowerGrad, int, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) : float_functor(attr0, attr1) {}
  BinaryFunctor<DeviceType::kCPU, BinaryOp::kScalarExpPowerGrad, float, float> float_functor;

  OF_DEVICE_FUNC Dst operator()(int src0, int src1) const {
    return static_cast<Dst>(float_functor(static_cast<float>(src0), static_cast<float>(src1)));
  }
};

template<typename Dst>
struct BinaryFunctor<DeviceType::kCPU, BinaryOp::kScalarExpPowerGrad, int8_t, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) : float_functor(attr0, attr1) {}
  BinaryFunctor<DeviceType::kCPU, BinaryOp::kScalarExpPowerGrad, float, float> float_functor;

  OF_DEVICE_FUNC Dst operator()(int8_t src0, int8_t src1) const {
    return static_cast<Dst>(float_functor(static_cast<float>(src0), static_cast<float>(src1)));
  }
};

template<typename Dst>
struct BinaryFunctor<DeviceType::kCPU, BinaryOp::kScalarExpPowerGrad, uint8_t, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) : float_functor(attr0, attr1) {}
  BinaryFunctor<DeviceType::kCPU, BinaryOp::kScalarExpPowerGrad, float, float> float_functor;

  OF_DEVICE_FUNC Dst operator()(uint8_t src0, uint8_t src1) const {
    return static_cast<Dst>(float_functor(static_cast<float>(src0), static_cast<float>(src1)));
  }
};

template<typename Dst>
struct BinaryFunctor<DeviceType::kCPU, BinaryOp::kScalarExpPowerGrad, int64_t, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) : float_functor(attr0, attr1) {}
  BinaryFunctor<DeviceType::kCPU, BinaryOp::kScalarExpPowerGrad, float, float> float_functor;

  OF_DEVICE_FUNC Dst operator()(int src0, int src1) const {
    return static_cast<Dst>(float_functor(static_cast<float>(src0), static_cast<float>(src1)));
  }
};

template<typename Dst>
struct BinaryFunctor<DeviceType::kCPU, BinaryOp::kScalarExpPowerGrad, float16, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) : scalar_operand(attr0.Value<float>()) {}

  OF_DEVICE_FUNC Dst operator()(float16 src0, float16 src1) const {
    return static_cast<Dst>(std::pow(scalar_operand, static_cast<float>(src0))
                            * std::log(scalar_operand) * static_cast<float>(src1));
  }
  float scalar_operand;
};

template<typename Src, typename Dst>
struct BinaryFunctor<DeviceType::kCPU, BinaryOp::kGeluBackwardWithDyX, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src dy, Src x) const {
    return static_cast<Dst>(
        0.5 * (1.0 + std::erf(inv_sqrt2 * x) + x * coef * std::exp(-0.5 * x * x)) * dy);
  }

  Src inv_sqrt2 = std::sqrt(0.5);
  Src coef = std::sqrt(2.0 / std::acos(-1.0));
};

template<typename Src, typename Dst>
struct BinaryFunctor<DeviceType::kCPU, BinaryOp::kFastGeluBackwardWithDyX, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src dy, Src x) const {
    // ref to: https://mlfromscratch.com/activation-functions-explained/#gelu
    const Src one = static_cast<Src>(1);
    const Src half = static_cast<Src>(0.5);
    const Src pow3 = x * x * x;
    const Src tanh_out = std::tanh(alpha * (x + beta * pow3));
    const Src dtanh = alpha * (half * x + beta * static_cast<Src>(1.5) * pow3);
    return dy * (half + half * tanh_out + dtanh * (one - tanh_out * tanh_out));
  }

 private:
  static constexpr Src alpha = static_cast<Src>(0.7978845608028654);
  static constexpr Src beta = static_cast<Src>(0.044714998453855515);
};

template<typename Src, typename Dst>
struct BinaryFunctor<DeviceType::kCPU, BinaryOp::kQuickGeluBackwardWithDyX, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src dy, Src x) const {
    const Src one = static_cast<Src>(1.0);
    const Src sigmoid = one / (one + exp(-x * alpha));
    return dy * (sigmoid + alpha * x * (sigmoid * (one - sigmoid)));
  }

 private:
  static constexpr Src alpha = static_cast<Src>(1.702);
};

template<typename Src, typename Dst>
struct BinaryFunctor<DeviceType::kCPU, BinaryOp::kTanhBackwardWithDyY, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src dy, Src y) const {
    return static_cast<Dst>(dy * (static_cast<Src>(1.0) - y * y));
  }
};

template<typename Src, typename Dst>
struct BinaryFunctor<DeviceType::kCPU, BinaryOp::kAcosBackwardWithDyX, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}
  OF_DEVICE_FUNC Dst operator()(Src dy, Src x) const {
    return dy * -(static_cast<Src>(1.0) / sqrt(static_cast<Src>(1.0) - x * x));
  }
};

template<typename Src, typename Dst>
struct BinaryFunctor<DeviceType::kCPU, BinaryOp::kAcoshBackwardWithDyX, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}
  OF_DEVICE_FUNC Dst operator()(Src dy, Src x) const {
    return dy / sqrt(x * x - static_cast<Src>(1.0));
  }
};

template<typename Src, typename Dst>
struct BinaryFunctor<DeviceType::kCPU, BinaryOp::kAsinBackwardWithDyX, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}
  OF_DEVICE_FUNC Dst operator()(Src dy, Src x) const {
    return dy * (static_cast<Src>(1.0) / sqrt(static_cast<Src>(1.0) - x * x));
  }
};

template<typename Src, typename Dst>
struct BinaryFunctor<DeviceType::kCPU, BinaryOp::kAsinhBackwardWithDyX, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}
  OF_DEVICE_FUNC Dst operator()(Src dy, Src x) const {
    return dy * (static_cast<Src>(1.0) / sqrt(static_cast<Src>(1.0) + x * x));
  }
};

template<typename Src, typename Dst>
struct BinaryFunctor<DeviceType::kCPU, BinaryOp::kErfBackwardWithDyX, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}
  OF_DEVICE_FUNC Dst operator()(Src dy, Src x) const {
    return dy * static_cast<Src>(2.0) * (static_cast<Src>(1.0) / sqrt(static_cast<Src>(M_PI)))
           * exp(-x * x);
  }
};

template<typename Src, typename Dst>
struct BinaryFunctor<DeviceType::kCPU, BinaryOp::kErfcBackwardWithDyX, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}
  OF_DEVICE_FUNC Dst operator()(Src dy, Src x) const {
    return dy * static_cast<Src>(-2.0) * (static_cast<Src>(1.0) / sqrt(static_cast<Src>(M_PI)))
           * exp(-x * x);
  }
};

template<>
struct BinaryFunctor<DeviceType::kCPU, BinaryOp::kDigammaBackwardWithDyX, float, float> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}
  OF_DEVICE_FUNC float operator()(float dy, float x) const {
    ep::primitive::UnaryFunctor<DeviceType::kCPU, UnaryOp::kTrigamma, float, float>
        trigamma_functor(0, 0);
    float trigamma_result = trigamma_functor(x);
    return trigamma_result * dy;
  }
};

template<>
struct BinaryFunctor<DeviceType::kCPU, BinaryOp::kDigammaBackwardWithDyX, double, double> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}
  OF_DEVICE_FUNC double operator()(double dy, double x) const {
    ep::primitive::UnaryFunctor<DeviceType::kCPU, UnaryOp::kTrigamma, double, double>
        trigamma_functor(0, 0);
    double trigamma_result = trigamma_functor(x);
    return trigamma_result * dy;
  }
};

template<typename Src, typename Dst>
struct BinaryFunctor<DeviceType::kCPU, BinaryOp::kLgammaBackwardWithDyX, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}
  OF_DEVICE_FUNC Dst operator()(Src dy, Src x) const {
    ep::primitive::UnaryFunctor<DeviceType::kCPU, UnaryOp::kDigamma, Src, Dst> digamma_functor(0,
                                                                                               0);
    Dst digamma_result = digamma_functor(x);
    return digamma_result * dy;
  }
};

template<typename Src, typename Dst>
struct BinaryFunctor<DeviceType::kCPU, BinaryOp::kZeta, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}
  OF_DEVICE_FUNC Dst operator()(Src x, Src q) const {
    // ref
    // https://github.com/pytorch/pytorch/blob/release/1.13/aten/src/ATen/native/Math.h#L235-L309
    const Src MACHEP = Src{1.11022302462515654042E-16};
    constexpr Src zero = Src{0.0};
    constexpr Src half = Src{0.5};
    constexpr Src one = Src{1.0};
    static const Src A[] = {
        12.0,
        -720.0,
        30240.0,
        -1209600.0,
        47900160.0,
        -1.8924375803183791606e9, /*1.307674368e12/691*/
        7.47242496e10,
        -2.950130727918164224e12,  /*1.067062284288e16/3617*/
        1.1646782814350067249e14,  /*5.109094217170944e18/43867*/
        -4.5979787224074726105e15, /*8.028576626982912e20/174611*/
        1.8152105401943546773e17,  /*1.5511210043330985984e23/854513*/
        -7.1661652561756670113e18  /*1.6938241367317436694528e27/236364091*/
    };
    int i = 0;
    Src a, b, k, s, t, w;
    if (x == one) { return std::numeric_limits<Dst>::infinity(); }

    if (x < one) { return std::numeric_limits<Dst>::quiet_NaN(); }

    if (q <= zero) {
      if (q == floor(q)) { return std::numeric_limits<Dst>::infinity(); }
      if (x != floor(x)) { return std::numeric_limits<Dst>::quiet_NaN(); }
    }

    s = pow(q, -x);
    a = q;
    i = 0;
    b = zero;
    while ((i < 9) || (a <= Src{9.0})) {
      i += 1;
      a += one;
      b = pow(a, -x);
      s += b;
      if ((-MACHEP * s < b) && (b < MACHEP * s)) { return static_cast<Dst>(s); }
    };

    w = a;
    s += b * w / (x - one);
    s -= half * b;
    a = one;
    k = zero;
    for (int i = 0; i < 12; i++) {
      a *= x + k;
      b /= w;
      t = a * b / A[i];
      s = s + t;
      t = fabs(t / s);
      if (t < MACHEP) { return static_cast<Dst>(s); }
      k += one;
      a *= x + k;
      b /= w;
      k += one;
    }
    return static_cast<Dst>(s);
  }
};

#define SPECIALIZATION_CPU_BINARY_FUNCTOR(op, type)                                          \
  template<>                                                                                 \
  struct BinaryFunctor<DeviceType::kCPU, op, type, type> {                                   \
    OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) : int_functor(attr0, attr1) {}  \
                                                                                             \
    BinaryFunctor<DeviceType::kCPU, op, int, int> int_functor;                               \
    OF_DEVICE_FUNC type operator()(type src0, type src1) const {                             \
      return static_cast<type>(int_functor(static_cast<int>(src0), static_cast<int>(src1))); \
    }                                                                                        \
  };

SPECIALIZATION_CPU_BINARY_FUNCTOR(BinaryOp::kPow, bool);
SPECIALIZATION_CPU_BINARY_FUNCTOR(BinaryOp::kFmod, bool);
SPECIALIZATION_CPU_BINARY_FUNCTOR(BinaryOp::kFloorDiv, bool);
SPECIALIZATION_CPU_BINARY_FUNCTOR(BinaryOp::kTruncDiv, bool);
SPECIALIZATION_CPU_BINARY_FUNCTOR(BinaryOp::kFloorMod, bool);
SPECIALIZATION_CPU_BINARY_FUNCTOR(BinaryOp::kScalarBasePowerGrad, bool);
SPECIALIZATION_CPU_BINARY_FUNCTOR(BinaryOp::kScalarExpPowerGrad, bool);
SPECIALIZATION_CPU_BINARY_FUNCTOR(BinaryOp::kPow, char);
SPECIALIZATION_CPU_BINARY_FUNCTOR(BinaryOp::kFmod, char);
SPECIALIZATION_CPU_BINARY_FUNCTOR(BinaryOp::kFloorDiv, char);
SPECIALIZATION_CPU_BINARY_FUNCTOR(BinaryOp::kTruncDiv, char);
SPECIALIZATION_CPU_BINARY_FUNCTOR(BinaryOp::kFloorMod, char);
SPECIALIZATION_CPU_BINARY_FUNCTOR(BinaryOp::kScalarBasePowerGrad, char);
SPECIALIZATION_CPU_BINARY_FUNCTOR(BinaryOp::kScalarExpPowerGrad, char);

}  // namespace broadcast_elementwise_binary
}  // namespace primitive
}  // namespace ep
}  // namespace oneflow
