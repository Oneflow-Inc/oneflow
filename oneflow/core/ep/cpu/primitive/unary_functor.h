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
  UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const {
    return static_cast<Src>(0.5) * src * (static_cast<Src>(1.0) + std::erf(inv_sqrt2 * src));
  }
  Src inv_sqrt2 = std::sqrt(0.5);
};

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCPU, UnaryOp::kTanh, Dst, Src> {
  UnaryFunctor(Scalar attr0, Scalar attr1) {}

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

}  // namespace primitive
}  // namespace ep
}  // namespace oneflow
