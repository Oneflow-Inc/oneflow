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
struct BinaryFunctor<DeviceType::kCPU, BinaryOp::kFloorMod, float, float> {
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
struct BinaryFunctor<DeviceType::kCPU, BinaryOp::kFloorMod, double, double> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC double operator()(double src0, double src1) const {
    double trunc_mod = fmod(src0, src1);
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
struct BinaryFunctor<DeviceType::kCPU, BinaryOp::kTanhBackwardWithDyX, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src dy, Src x) const {
    Src tanh_val = std::tanh(x);
    return static_cast<Dst>(dy * (static_cast<Src>(1.0) - tanh_val * tanh_val));
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
SPECIALIZATION_CPU_BINARY_FUNCTOR(BinaryOp::kFloorMod, bool);
SPECIALIZATION_CPU_BINARY_FUNCTOR(BinaryOp::kPow, char);
SPECIALIZATION_CPU_BINARY_FUNCTOR(BinaryOp::kFmod, char);
SPECIALIZATION_CPU_BINARY_FUNCTOR(BinaryOp::kFloorDiv, char);
SPECIALIZATION_CPU_BINARY_FUNCTOR(BinaryOp::kFloorMod, char);

}  // namespace broadcast_elementwise_binary
}  // namespace primitive
}  // namespace ep
}  // namespace oneflow
