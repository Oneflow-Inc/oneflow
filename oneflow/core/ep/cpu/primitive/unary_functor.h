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

}  // namespace primitive
}  // namespace ep
}  // namespace oneflow
