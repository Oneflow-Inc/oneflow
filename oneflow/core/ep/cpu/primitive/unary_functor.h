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
#include <cmath>

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
struct UnaryFunctor<DeviceType::kCPU, UnaryOp::kTanh, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const { return std::tanh(src); }
};

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCPU, UnaryOp::kRsqrt, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const { return static_cast<Src>(1.0) / std::sqrt(src); }
};

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCPU, UnaryOp::kAcos, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const { return std::acos(src); }
};

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCPU, UnaryOp::kAcosh, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const { return std::acosh(src); }
};

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCPU, UnaryOp::kAsin, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const { return std::asin(src); }
};

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCPU, UnaryOp::kAsinh, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const { return std::asinh(src); }
};

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCPU, UnaryOp::kAtan, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const { return std::atan(src); }
};

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCPU, UnaryOp::kAtanh, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const { return std::atanh(src); }
};

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCPU, UnaryOp::kCeil, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const { return std::ceil(src); }
};

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCPU, UnaryOp::kCos, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const { return std::cos(src); }
};

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCPU, UnaryOp::kCosh, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const { return std::cosh(src); }
};

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCPU, UnaryOp::kErf, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const { return std::erf(src); }
};

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCPU, UnaryOp::kErfc, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const { return std::erfc(src); }
};

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCPU, UnaryOp::kExp, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const { return std::exp(src); }
};

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCPU, UnaryOp::kExpm1, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const { return std::expm1(src); }
};

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCPU, UnaryOp::kFloor, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const { return std::floor(src); }
};

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCPU, UnaryOp::kLgamma, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const { return std::lgamma(src); }
};

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCPU, UnaryOp::kLog, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const { return std::log(src); }
};

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCPU, UnaryOp::kLog2, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const { return std::log2(src); }
};

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCPU, UnaryOp::kLog1p, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const { return std::log1p(src); }
};

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCPU, UnaryOp::kRint, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const { return std::rint(src); }
};

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCPU, UnaryOp::kRound, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const { return std::nearbyint(src); }
};

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCPU, UnaryOp::kSin, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const { return std::sin(src); }
};

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCPU, UnaryOp::kSinh, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const { return std::sinh(src); }
};

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCPU, UnaryOp::kSqrt, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const { return std::sqrt(src); }
};

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCPU, UnaryOp::kTan, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const { return std::tan(src); }
};

template<>
struct UnaryFunctor<DeviceType::kCPU, UnaryOp::kIsInf, bool, float> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC bool operator()(float src) const { return std::isinf(src); }
};

template<>
struct UnaryFunctor<DeviceType::kCPU, UnaryOp::kIsInf, bool, double> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC bool operator()(double src) const { return std::isinf(src); }
};

template<>
struct UnaryFunctor<DeviceType::kCPU, UnaryOp::kIsNan, bool, float> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC bool operator()(float src) const { return std::isnan(src); }
};

template<>
struct UnaryFunctor<DeviceType::kCPU, UnaryOp::kIsNan, bool, double> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC bool operator()(double src) const { return std::isnan(src); }
};

template<typename Src>
struct UnaryFunctor<DeviceType::kCPU, UnaryOp::kIsFinite, bool, Src> {
  UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC bool operator()(Src src) const { return std::isfinite(src); }
};

}  // namespace primitive
}  // namespace ep
}  // namespace oneflow
