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

namespace oneflow {
namespace ep {
namespace primitive {
namespace broadcast_elementwise_binary {

template<typename Src, typename Dst>
struct BinaryFunctor<DeviceType::kCUDA, BinaryOp::kPow, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src0, Src src1) const { return pow(src0, src1); }
};

template<>
struct BinaryFunctor<DeviceType::kCUDA, BinaryOp::kFmod, float, float> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC float operator()(float src0, float src1) const { return fmod(src0, src1); }
};

template<>
struct BinaryFunctor<DeviceType::kCUDA, BinaryOp::kFmod, double, double> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC double operator()(double src0, double src1) const { return fmod(src0, src1); }
};

template<>
struct BinaryFunctor<DeviceType::kCUDA, BinaryOp::kFloorDiv, float, float> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC float operator()(float src0, float src1) const { return floor(src0 / src1); }
};

template<>
struct BinaryFunctor<DeviceType::kCUDA, BinaryOp::kFloorDiv, double, double> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC double operator()(double src0, double src1) const { return floor(src0 / src1); }
};

template<>
struct BinaryFunctor<DeviceType::kCUDA, BinaryOp::kTruncDiv, float, float> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC float operator()(float src0, float src1) const { return truncf(src0 / src1); }
};

template<>
struct BinaryFunctor<DeviceType::kCUDA, BinaryOp::kTruncDiv, double, double> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC double operator()(double src0, double src1) const { return trunc(src0 / src1); }
};

template<>
struct BinaryFunctor<DeviceType::kCUDA, BinaryOp::kFloorMod, float, float> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC float operator()(float src0, float src1) const {
    float trunc_mod = fmod(src0, src1);
    return (trunc_mod != static_cast<float>(0))
                   && ((src1 < static_cast<float>(0)) != (trunc_mod < static_cast<float>(0)))
               ? trunc_mod + src1
               : trunc_mod;
  }
};

template<>
struct BinaryFunctor<DeviceType::kCUDA, BinaryOp::kFloorMod, double, double> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC double operator()(double src0, double src1) const {
    double trunc_mod = fmod(src0, src1);
    return (trunc_mod != static_cast<double>(0))
                   && ((src1 < static_cast<double>(0)) != (trunc_mod < static_cast<double>(0)))
               ? trunc_mod + src1
               : trunc_mod;
  }
};

template<typename Src, typename Dst>
struct BinaryFunctor<DeviceType::kCUDA, BinaryOp::kGeluBackwardWithDyX, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {
#if defined(__CUDA_ARCH__)
    coef = sqrt(static_cast<Src>(2.0) / acos(static_cast<Src>(-1.0)));
#else
    coef = std::sqrt(static_cast<Src>(2.0) / std::acos(static_cast<Src>(-1.0)));
#endif
  }

  OF_DEVICE_FUNC Dst operator()(Src dy, Src x) const {
    return static_cast<Src>(0.5)
           * (static_cast<Src>(1.0) + erf(static_cast<Src>(M_SQRT1_2) * x)
              + x * coef * exp(static_cast<Src>(-0.5) * x * x))
           * dy;
  }
  Src coef;
};

template<typename Src, typename Dst>
struct BinaryFunctor<DeviceType::kCUDA, BinaryOp::kFastGeluBackwardWithDyX, Src, Dst> {
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
  const Src alpha = static_cast<Src>(0.7978845608028654);
  const Src beta = static_cast<Src>(0.044714998453855515);
};

template<typename Src, typename Dst>
struct BinaryFunctor<DeviceType::kCUDA, BinaryOp::kQuickGeluBackwardWithDyX, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src dy, Src x) const {
    const Src one = static_cast<Src>(1.0);
    const Src sigmoid = one / (one + exp(-x * alpha));
    return dy * (sigmoid + alpha * x * (sigmoid * (one - sigmoid)));
  }

 private:
  const Src alpha = static_cast<Src>(1.702);
};

template<typename Src, typename Dst>
struct BinaryFunctor<DeviceType::kCUDA, BinaryOp::kTanhBackwardWithDyX, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src dy, Src x) const {
    Src tanh_val = tanh(x);
    return static_cast<Dst>(dy * (static_cast<Src>(1.0) - tanh_val * tanh_val));
  }
};

template<typename Dst>
struct BinaryFunctor<DeviceType::kCUDA, BinaryOp::kScalarExpPowerGrad, int, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) : float_functor(attr0, attr1) {}
  BinaryFunctor<DeviceType::kCUDA, BinaryOp::kScalarExpPowerGrad, float, float> float_functor;

  OF_DEVICE_FUNC Dst operator()(int src0, int src1) const {
    return static_cast<Dst>(float_functor(static_cast<float>(src0), static_cast<float>(src1)));
  }
};

template<typename Dst>
struct BinaryFunctor<DeviceType::kCUDA, BinaryOp::kScalarExpPowerGrad, int8_t, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) : float_functor(attr0, attr1) {}
  BinaryFunctor<DeviceType::kCUDA, BinaryOp::kScalarExpPowerGrad, float, float> float_functor;

  OF_DEVICE_FUNC Dst operator()(int8_t src0, int8_t src1) const {
    return static_cast<Dst>(float_functor(static_cast<float>(src0), static_cast<float>(src1)));
  }
};

template<typename Dst>
struct BinaryFunctor<DeviceType::kCUDA, BinaryOp::kScalarExpPowerGrad, uint8_t, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) : float_functor(attr0, attr1) {}
  BinaryFunctor<DeviceType::kCUDA, BinaryOp::kScalarExpPowerGrad, float, float> float_functor;

  OF_DEVICE_FUNC Dst operator()(uint8_t src0, uint8_t src1) const {
    return static_cast<Dst>(float_functor(static_cast<float>(src0), static_cast<float>(src1)));
  }
};

template<typename Dst>
struct BinaryFunctor<DeviceType::kCUDA, BinaryOp::kScalarExpPowerGrad, int64_t, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) : float_functor(attr0, attr1) {}
  BinaryFunctor<DeviceType::kCUDA, BinaryOp::kScalarExpPowerGrad, float, float> float_functor;

  OF_DEVICE_FUNC Dst operator()(int src0, int src1) const {
    return static_cast<Dst>(float_functor(static_cast<float>(src0), static_cast<float>(src1)));
  }
};

template<typename Src, typename Dst>
struct BinaryFunctor<DeviceType::kCUDA, BinaryOp::kAtanhBackwardWithDyX, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}
  OF_DEVICE_FUNC Dst operator()(Src dy, Src x) const {
    const Src one = static_cast<Src>(1.0);
    return dy * one / (one - static_cast<Src>(pow(x, 2)));
  }
};

template<typename Src, typename Dst>
struct BinaryFunctor<DeviceType::kCUDA, BinaryOp::kIsCloseEqualNan, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1)
      : atol(attr0.Value<float>()), rtol(attr1.Value<float>()) {}

  OF_DEVICE_FUNC Dst operator()(Src src0, Src src1) const {
    bool close = src0 == src1;
    close |= (isnan(src0) and isnan(src1));
    if (atol == 0 and rtol == 0) return close;
    Src allowed_error = static_cast<Src>(atol) + abs(static_cast<Src>(rtol) * src1);
    Src actual_error = abs(src0 - src1);
    close |= (isfinite(actual_error) and (actual_error <= allowed_error));
    return close;
  }
  float atol, rtol;
};

template<typename Src, typename Dst>
struct BinaryFunctor<DeviceType::kCUDA, BinaryOp::kIsClose, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1)
      : atol(attr0.Value<float>()), rtol(attr1.Value<float>()) {}

  OF_DEVICE_FUNC Dst operator()(Src src0, Src src1) const {
    bool close = src0 == src1;
    if (atol == 0 and rtol == 0) return close;
    Src allowed_error = static_cast<Src>(atol) + abs(static_cast<Src>(rtol) * src1);
    Src actual_error = abs(src0 - src1);
    close |= (isfinite(actual_error) and (actual_error <= allowed_error));
    return close;
  }
  float atol, rtol;
};

#define SPECIALIZATION_INTEGRAL_CLOSENESS_BINARY_FUNCTOR(op, type)                            \
  template<typename Dst>                                                                      \
  struct BinaryFunctor<DeviceType::kCUDA, op, type, Dst> {                                    \
    OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) : float_functor(attr0, attr1) {} \
    OF_DEVICE_FUNC Dst operator()(type src0, type src1) const {                               \
      return float_functor(static_cast<float>(src0), static_cast<float>(src1));               \
    }                                                                                         \
    BinaryFunctor<DeviceType::kCUDA, op, float, Dst> float_functor;                           \
  };
SPECIALIZATION_INTEGRAL_CLOSENESS_BINARY_FUNCTOR(BinaryOp::kIsClose, bool);
SPECIALIZATION_INTEGRAL_CLOSENESS_BINARY_FUNCTOR(BinaryOp::kIsClose, int);
SPECIALIZATION_INTEGRAL_CLOSENESS_BINARY_FUNCTOR(BinaryOp::kIsClose, char);
SPECIALIZATION_INTEGRAL_CLOSENESS_BINARY_FUNCTOR(BinaryOp::kIsClose, int8_t);
SPECIALIZATION_INTEGRAL_CLOSENESS_BINARY_FUNCTOR(BinaryOp::kIsClose, uint8_t);
SPECIALIZATION_INTEGRAL_CLOSENESS_BINARY_FUNCTOR(BinaryOp::kIsClose, int64_t);
SPECIALIZATION_INTEGRAL_CLOSENESS_BINARY_FUNCTOR(BinaryOp::kIsCloseEqualNan, bool);
SPECIALIZATION_INTEGRAL_CLOSENESS_BINARY_FUNCTOR(BinaryOp::kIsCloseEqualNan, int);
SPECIALIZATION_INTEGRAL_CLOSENESS_BINARY_FUNCTOR(BinaryOp::kIsCloseEqualNan, char);
SPECIALIZATION_INTEGRAL_CLOSENESS_BINARY_FUNCTOR(BinaryOp::kIsCloseEqualNan, int8_t);
SPECIALIZATION_INTEGRAL_CLOSENESS_BINARY_FUNCTOR(BinaryOp::kIsCloseEqualNan, uint8_t);
SPECIALIZATION_INTEGRAL_CLOSENESS_BINARY_FUNCTOR(BinaryOp::kIsCloseEqualNan, int64_t);

/*********nv_bfloat16_kernel*******/

#if CUDA_VERSION >= 11000

#define SPECIALIZATION_PSEUDO_BFLOAT16_BINARY_FUNCTOR(op)                                     \
  template<>                                                                                  \
  struct BinaryFunctor<DeviceType::kCUDA, op, nv_bfloat16, nv_bfloat16> {                     \
    OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) : float_functor(attr0, attr1) {} \
                                                                                              \
    BinaryFunctor<DeviceType::kCUDA, op, float, float> float_functor;                         \
    OF_DEVICE_FUNC nv_bfloat16 operator()(nv_bfloat16 src0, nv_bfloat16 src1) const {         \
      return __float2bfloat16(float_functor(__bfloat162float(src0), __bfloat162float(src1))); \
    }                                                                                         \
  };

SPECIALIZATION_PSEUDO_BFLOAT16_BINARY_FUNCTOR(BinaryOp::kPow);
SPECIALIZATION_PSEUDO_BFLOAT16_BINARY_FUNCTOR(BinaryOp::kFmod);
SPECIALIZATION_PSEUDO_BFLOAT16_BINARY_FUNCTOR(BinaryOp::kFloorDiv);
SPECIALIZATION_PSEUDO_BFLOAT16_BINARY_FUNCTOR(BinaryOp::kTruncDiv);
SPECIALIZATION_PSEUDO_BFLOAT16_BINARY_FUNCTOR(BinaryOp::kFloorMod);
SPECIALIZATION_PSEUDO_BFLOAT16_BINARY_FUNCTOR(BinaryOp::kScalarBasePowerGrad);
SPECIALIZATION_PSEUDO_BFLOAT16_BINARY_FUNCTOR(BinaryOp::kScalarExpPowerGrad);
SPECIALIZATION_PSEUDO_BFLOAT16_BINARY_FUNCTOR(BinaryOp::kIdentityBackwardWithDyX);
SPECIALIZATION_PSEUDO_BFLOAT16_BINARY_FUNCTOR(BinaryOp::kEluBackwardWithDyX);
SPECIALIZATION_PSEUDO_BFLOAT16_BINARY_FUNCTOR(BinaryOp::kCeluBackwardWithDyY);
SPECIALIZATION_PSEUDO_BFLOAT16_BINARY_FUNCTOR(BinaryOp::kGeluBackwardWithDyX);
SPECIALIZATION_PSEUDO_BFLOAT16_BINARY_FUNCTOR(BinaryOp::kHardswishBackwardWithDyX);
SPECIALIZATION_PSEUDO_BFLOAT16_BINARY_FUNCTOR(BinaryOp::kHardsigmoidBackwardWithDyX);
SPECIALIZATION_PSEUDO_BFLOAT16_BINARY_FUNCTOR(BinaryOp::kHardshrinkBackwardWithDyY);
SPECIALIZATION_PSEUDO_BFLOAT16_BINARY_FUNCTOR(BinaryOp::kHardtanhBackwardWithDyY);
SPECIALIZATION_PSEUDO_BFLOAT16_BINARY_FUNCTOR(BinaryOp::kLeakyReluBackwardWithDyX);
SPECIALIZATION_PSEUDO_BFLOAT16_BINARY_FUNCTOR(BinaryOp::kMishBackwardWithDyX);
SPECIALIZATION_PSEUDO_BFLOAT16_BINARY_FUNCTOR(BinaryOp::kSeluBackwardWithDyX);
SPECIALIZATION_PSEUDO_BFLOAT16_BINARY_FUNCTOR(BinaryOp::kSiluBackwardWithDyX);
SPECIALIZATION_PSEUDO_BFLOAT16_BINARY_FUNCTOR(BinaryOp::kSoftsignBackwardWithDyX);
SPECIALIZATION_PSEUDO_BFLOAT16_BINARY_FUNCTOR(BinaryOp::kSoftplusBackwardWithDyX);
SPECIALIZATION_PSEUDO_BFLOAT16_BINARY_FUNCTOR(BinaryOp::kSoftshrinkBackwardWithDyY);
SPECIALIZATION_PSEUDO_BFLOAT16_BINARY_FUNCTOR(BinaryOp::kTanhBackwardWithDyX);
SPECIALIZATION_PSEUDO_BFLOAT16_BINARY_FUNCTOR(BinaryOp::kThresholdBackwardWithDyX);
SPECIALIZATION_PSEUDO_BFLOAT16_BINARY_FUNCTOR(BinaryOp::kFastGeluBackwardWithDyX);
SPECIALIZATION_PSEUDO_BFLOAT16_BINARY_FUNCTOR(BinaryOp::kQuickGeluBackwardWithDyX);

SPECIALIZATION_PSEUDO_BFLOAT16_BINARY_FUNCTOR(BinaryOp::kAcosBackwardWithDyX);
SPECIALIZATION_PSEUDO_BFLOAT16_BINARY_FUNCTOR(BinaryOp::kAcoshBackwardWithDyX);
SPECIALIZATION_PSEUDO_BFLOAT16_BINARY_FUNCTOR(BinaryOp::kAsinBackwardWithDyX);
SPECIALIZATION_PSEUDO_BFLOAT16_BINARY_FUNCTOR(BinaryOp::kAsinhBackwardWithDyX);
SPECIALIZATION_PSEUDO_BFLOAT16_BINARY_FUNCTOR(BinaryOp::kCosBackwardWithDyX);
SPECIALIZATION_PSEUDO_BFLOAT16_BINARY_FUNCTOR(BinaryOp::kCoshBackwardWithDyX);
SPECIALIZATION_PSEUDO_BFLOAT16_BINARY_FUNCTOR(BinaryOp::kErfBackwardWithDyX);
SPECIALIZATION_PSEUDO_BFLOAT16_BINARY_FUNCTOR(BinaryOp::kErfcBackwardWithDyX);
SPECIALIZATION_PSEUDO_BFLOAT16_BINARY_FUNCTOR(BinaryOp::kExpBackwardWithDyX);
SPECIALIZATION_PSEUDO_BFLOAT16_BINARY_FUNCTOR(BinaryOp::kExp2BackwardWithDyX);
SPECIALIZATION_PSEUDO_BFLOAT16_BINARY_FUNCTOR(BinaryOp::kExpm1BackwardWithDyX);
SPECIALIZATION_PSEUDO_BFLOAT16_BINARY_FUNCTOR(BinaryOp::kLog2BackwardWithDyX);
SPECIALIZATION_PSEUDO_BFLOAT16_BINARY_FUNCTOR(BinaryOp::kLog10BackwardWithDyX);
SPECIALIZATION_PSEUDO_BFLOAT16_BINARY_FUNCTOR(BinaryOp::kLogSigmoidBackwardWithDyX);
SPECIALIZATION_PSEUDO_BFLOAT16_BINARY_FUNCTOR(BinaryOp::kReciprocalNoNanBackwardWithDyX);
SPECIALIZATION_PSEUDO_BFLOAT16_BINARY_FUNCTOR(BinaryOp::kRsqrtBackwardWithDyX);
SPECIALIZATION_PSEUDO_BFLOAT16_BINARY_FUNCTOR(BinaryOp::kSinBackwardWithDyX);
SPECIALIZATION_PSEUDO_BFLOAT16_BINARY_FUNCTOR(BinaryOp::kSinhBackwardWithDyX);
SPECIALIZATION_PSEUDO_BFLOAT16_BINARY_FUNCTOR(BinaryOp::kSqrtBackwardWithDyX);
SPECIALIZATION_PSEUDO_BFLOAT16_BINARY_FUNCTOR(BinaryOp::kTanBackwardWithDyX);
SPECIALIZATION_PSEUDO_BFLOAT16_BINARY_FUNCTOR(BinaryOp::kSigmoidBackwardWithDyY);
SPECIALIZATION_PSEUDO_BFLOAT16_BINARY_FUNCTOR(BinaryOp::kSigmoidBackwardWithDyX);
SPECIALIZATION_PSEUDO_BFLOAT16_BINARY_FUNCTOR(BinaryOp::kAtanhBackwardWithDyX);

#define SPECIALIZATION_BFLOAT16_COMPARISON_BINARY_FUNCTOR(op)                                 \
  template<typename Dst>                                                                      \
  struct BinaryFunctor<DeviceType::kCUDA, op, nv_bfloat16, Dst> {                             \
    OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) : float_functor(attr0, attr1) {} \
    BinaryFunctor<DeviceType::kCUDA, op, float, Dst> float_functor;                           \
    OF_DEVICE_FUNC Dst operator()(nv_bfloat16 src0, nv_bfloat16 src1) const {                 \
      return float_functor(__bfloat162float(src0), __bfloat162float(src1));                   \
    }                                                                                         \
  };
SPECIALIZATION_BFLOAT16_COMPARISON_BINARY_FUNCTOR(BinaryOp::kIsCloseEqualNan)
SPECIALIZATION_BFLOAT16_COMPARISON_BINARY_FUNCTOR(BinaryOp::kIsClose)

#endif  // CUDA_VERSION >= 11000

#define SPECIALIZATION_PSEUDO_HALF_BINARY_FUNCTOR(op)                                         \
  template<>                                                                                  \
  struct BinaryFunctor<DeviceType::kCUDA, op, half, half> {                                   \
    OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) : float_functor(attr0, attr1) {} \
                                                                                              \
    BinaryFunctor<DeviceType::kCUDA, op, float, float> float_functor;                         \
    OF_DEVICE_FUNC half operator()(half src0, half src1) const {                              \
      return __float2half(float_functor(__half2float(src0), __half2float(src1)));             \
    }                                                                                         \
  };

SPECIALIZATION_PSEUDO_HALF_BINARY_FUNCTOR(BinaryOp::kPow);
SPECIALIZATION_PSEUDO_HALF_BINARY_FUNCTOR(BinaryOp::kFmod);
SPECIALIZATION_PSEUDO_HALF_BINARY_FUNCTOR(BinaryOp::kFloorDiv);
SPECIALIZATION_PSEUDO_HALF_BINARY_FUNCTOR(BinaryOp::kTruncDiv);
SPECIALIZATION_PSEUDO_HALF_BINARY_FUNCTOR(BinaryOp::kFloorMod);
SPECIALIZATION_PSEUDO_HALF_BINARY_FUNCTOR(BinaryOp::kScalarBasePowerGrad);
SPECIALIZATION_PSEUDO_HALF_BINARY_FUNCTOR(BinaryOp::kScalarExpPowerGrad);
SPECIALIZATION_PSEUDO_HALF_BINARY_FUNCTOR(BinaryOp::kEluBackwardWithDyX);
SPECIALIZATION_PSEUDO_HALF_BINARY_FUNCTOR(BinaryOp::kCeluBackwardWithDyY);
SPECIALIZATION_PSEUDO_HALF_BINARY_FUNCTOR(BinaryOp::kGeluBackwardWithDyX);
SPECIALIZATION_PSEUDO_HALF_BINARY_FUNCTOR(BinaryOp::kHardswishBackwardWithDyX);
SPECIALIZATION_PSEUDO_HALF_BINARY_FUNCTOR(BinaryOp::kHardshrinkBackwardWithDyY);
SPECIALIZATION_PSEUDO_HALF_BINARY_FUNCTOR(BinaryOp::kMishBackwardWithDyX);
SPECIALIZATION_PSEUDO_HALF_BINARY_FUNCTOR(BinaryOp::kSiluBackwardWithDyX);
SPECIALIZATION_PSEUDO_HALF_BINARY_FUNCTOR(BinaryOp::kSeluBackwardWithDyX);
SPECIALIZATION_PSEUDO_HALF_BINARY_FUNCTOR(BinaryOp::kSoftplusBackwardWithDyX);
SPECIALIZATION_PSEUDO_HALF_BINARY_FUNCTOR(BinaryOp::kSoftsignBackwardWithDyX);
SPECIALIZATION_PSEUDO_HALF_BINARY_FUNCTOR(BinaryOp::kSoftshrinkBackwardWithDyY);
SPECIALIZATION_PSEUDO_HALF_BINARY_FUNCTOR(BinaryOp::kThresholdBackwardWithDyX);
SPECIALIZATION_PSEUDO_HALF_BINARY_FUNCTOR(BinaryOp::kTanhBackwardWithDyX);
SPECIALIZATION_PSEUDO_HALF_BINARY_FUNCTOR(BinaryOp::kFastGeluBackwardWithDyX);
SPECIALIZATION_PSEUDO_HALF_BINARY_FUNCTOR(BinaryOp::kQuickGeluBackwardWithDyX);

SPECIALIZATION_PSEUDO_HALF_BINARY_FUNCTOR(BinaryOp::kAcosBackwardWithDyX);
SPECIALIZATION_PSEUDO_HALF_BINARY_FUNCTOR(BinaryOp::kAcoshBackwardWithDyX);
SPECIALIZATION_PSEUDO_HALF_BINARY_FUNCTOR(BinaryOp::kAsinBackwardWithDyX);
SPECIALIZATION_PSEUDO_HALF_BINARY_FUNCTOR(BinaryOp::kAsinhBackwardWithDyX);
SPECIALIZATION_PSEUDO_HALF_BINARY_FUNCTOR(BinaryOp::kCosBackwardWithDyX);
SPECIALIZATION_PSEUDO_HALF_BINARY_FUNCTOR(BinaryOp::kCoshBackwardWithDyX);
SPECIALIZATION_PSEUDO_HALF_BINARY_FUNCTOR(BinaryOp::kErfBackwardWithDyX);
SPECIALIZATION_PSEUDO_HALF_BINARY_FUNCTOR(BinaryOp::kErfcBackwardWithDyX);
SPECIALIZATION_PSEUDO_HALF_BINARY_FUNCTOR(BinaryOp::kExpBackwardWithDyX);
SPECIALIZATION_PSEUDO_HALF_BINARY_FUNCTOR(BinaryOp::kExp2BackwardWithDyX);
SPECIALIZATION_PSEUDO_HALF_BINARY_FUNCTOR(BinaryOp::kExpm1BackwardWithDyX);
SPECIALIZATION_PSEUDO_HALF_BINARY_FUNCTOR(BinaryOp::kLog2BackwardWithDyX);
SPECIALIZATION_PSEUDO_HALF_BINARY_FUNCTOR(BinaryOp::kLog10BackwardWithDyX);
SPECIALIZATION_PSEUDO_HALF_BINARY_FUNCTOR(BinaryOp::kLogSigmoidBackwardWithDyX);
SPECIALIZATION_PSEUDO_HALF_BINARY_FUNCTOR(BinaryOp::kReciprocalNoNanBackwardWithDyX);
SPECIALIZATION_PSEUDO_HALF_BINARY_FUNCTOR(BinaryOp::kRsqrtBackwardWithDyX);
SPECIALIZATION_PSEUDO_HALF_BINARY_FUNCTOR(BinaryOp::kSinBackwardWithDyX);
SPECIALIZATION_PSEUDO_HALF_BINARY_FUNCTOR(BinaryOp::kSinhBackwardWithDyX);
SPECIALIZATION_PSEUDO_HALF_BINARY_FUNCTOR(BinaryOp::kSqrtBackwardWithDyX);
SPECIALIZATION_PSEUDO_HALF_BINARY_FUNCTOR(BinaryOp::kTanBackwardWithDyX);
SPECIALIZATION_PSEUDO_HALF_BINARY_FUNCTOR(BinaryOp::kSigmoidBackwardWithDyY);
SPECIALIZATION_PSEUDO_HALF_BINARY_FUNCTOR(BinaryOp::kSigmoidBackwardWithDyX);
SPECIALIZATION_PSEUDO_HALF_BINARY_FUNCTOR(BinaryOp::kAtanhBackwardWithDyX);

#define SPECIALIZATION_HALF_COMPARISON_BINARY_FUNCTOR(op)                                     \
  template<typename Dst>                                                                      \
  struct BinaryFunctor<DeviceType::kCUDA, op, half, Dst> {                                    \
    OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) : float_functor(attr0, attr1) {} \
    BinaryFunctor<DeviceType::kCUDA, op, float, Dst> float_functor;                           \
    OF_DEVICE_FUNC Dst operator()(half src0, half src1) const {                               \
      return float_functor(__half2float(src0), __half2float(src1));                           \
    }                                                                                         \
  };

SPECIALIZATION_HALF_COMPARISON_BINARY_FUNCTOR(BinaryOp::kIsCloseEqualNan)
SPECIALIZATION_HALF_COMPARISON_BINARY_FUNCTOR(BinaryOp::kIsClose)

#define SPECIALIZATION_GPU_BINARY_FUNCTOR(op, type)                                          \
  template<>                                                                                 \
  struct BinaryFunctor<DeviceType::kCUDA, op, type, type> {                                  \
    OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) : int_functor(attr0, attr1) {}  \
                                                                                             \
    BinaryFunctor<DeviceType::kCUDA, op, int, int> int_functor;                              \
    OF_DEVICE_FUNC type operator()(type src0, type src1) const {                             \
      return static_cast<type>(int_functor(static_cast<int>(src0), static_cast<int>(src1))); \
    }                                                                                        \
  };

SPECIALIZATION_GPU_BINARY_FUNCTOR(BinaryOp::kPow, bool);
SPECIALIZATION_GPU_BINARY_FUNCTOR(BinaryOp::kFmod, bool);
SPECIALIZATION_GPU_BINARY_FUNCTOR(BinaryOp::kFloorDiv, bool);
SPECIALIZATION_GPU_BINARY_FUNCTOR(BinaryOp::kTruncDiv, bool);
SPECIALIZATION_GPU_BINARY_FUNCTOR(BinaryOp::kFloorMod, bool);
SPECIALIZATION_GPU_BINARY_FUNCTOR(BinaryOp::kScalarBasePowerGrad, bool);
SPECIALIZATION_GPU_BINARY_FUNCTOR(BinaryOp::kScalarExpPowerGrad, bool);
SPECIALIZATION_GPU_BINARY_FUNCTOR(BinaryOp::kPow, char);
SPECIALIZATION_GPU_BINARY_FUNCTOR(BinaryOp::kFmod, char);
SPECIALIZATION_GPU_BINARY_FUNCTOR(BinaryOp::kFloorDiv, char);
SPECIALIZATION_GPU_BINARY_FUNCTOR(BinaryOp::kTruncDiv, char);
SPECIALIZATION_GPU_BINARY_FUNCTOR(BinaryOp::kFloorMod, char);
SPECIALIZATION_GPU_BINARY_FUNCTOR(BinaryOp::kScalarBasePowerGrad, char);
SPECIALIZATION_GPU_BINARY_FUNCTOR(BinaryOp::kScalarExpPowerGrad, char);

}  // namespace broadcast_elementwise_binary
}  // namespace primitive
}  // namespace ep
}  // namespace oneflow
