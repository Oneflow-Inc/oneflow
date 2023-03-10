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
#ifndef ONEFLOW_CORE_PRIMITIVE_COMMON_BINARY_FUNCTOR_H_
#define ONEFLOW_CORE_PRIMITIVE_COMMON_BINARY_FUNCTOR_H_

#include "oneflow/core/ep/include/primitive/binary_op.h"
#include "oneflow/core/ep/common/primitive/unary_functor.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/scalar.h"
#include <cmath>

namespace oneflow {

namespace ep {
namespace primitive {
namespace broadcast_elementwise_binary {

template<DeviceType device, BinaryOp binary_op, typename Src, typename Dst>
struct BinaryFunctor;

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kAdd, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src0, Src src1) const { return static_cast<Dst>(src0 + src1); }
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kSub, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src0, Src src1) const { return static_cast<Dst>(src0 - src1); }
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kMul, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src0, Src src1) const { return static_cast<Dst>(src0 * src1); }
};

template<DeviceType device>
struct BinaryFunctor<device, BinaryOp::kMul, bool, bool> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC bool operator()(bool src0, bool src1) const { return src0 && src1; }
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kDiv, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src0, Src src1) const { return static_cast<Dst>(src0 / src1); }
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kMax, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src0, Src src1) const {
    return static_cast<Dst>(src0 > src1 ? src0 : src1);
  }
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kMin, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src0, Src src1) const {
    return static_cast<Dst>(src0 < src1 ? src0 : src1);
  }
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kBitwiseAnd, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src0, Src src1) const { return static_cast<Dst>(src0 & src1); }
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kBitwiseOr, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src0, Src src1) const { return static_cast<Dst>(src0 | src1); }
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kBitwiseXor, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src0, Src src1) const { return static_cast<Dst>(src0 ^ src1); }
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kEqual, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src0, Src src1) const { return static_cast<Dst>(src0 == src1); }
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kNotEqual, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src0, Src src1) const { return static_cast<Dst>(src0 != src1); }
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kLessThan, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src0, Src src1) const { return static_cast<Dst>(src0 < src1); }
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kLessEqual, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src0, Src src1) const { return static_cast<Dst>(src0 <= src1); }
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kGreaterThan, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src0, Src src1) const { return static_cast<Dst>(src0 > src1); }
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kGreaterEqual, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src0, Src src1) const { return static_cast<Dst>(src0 >= src1); }
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kIsCloseEqualNan, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1)
      : atol(attr0.Value<float>()), rtol(attr1.Value<float>()) {}

  OF_DEVICE_FUNC Dst operator()(Src src0, Src src1) const {
    bool close = src0 == src1;
    close |= (std::isnan(src0) and std::isnan(src1));
    if (atol == 0 and rtol == 0) return close;
    Src allowed_error = static_cast<Src>(atol) + abs(static_cast<Src>(rtol) * src1);
    Src actual_error = abs(src0 - src1);
    close |= (std::isfinite(actual_error) and (actual_error <= allowed_error));
    return close;
  }
  float atol, rtol;
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kIsClose, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1)
      : atol(attr0.Value<float>()), rtol(attr1.Value<float>()) {}

  OF_DEVICE_FUNC Dst operator()(Src src0, Src src1) const {
    bool close = src0 == src1;
    if (atol == 0 and rtol == 0) return close;
    Src allowed_error = static_cast<Src>(atol) + abs(static_cast<Src>(rtol) * src1);
    Src actual_error = abs(src0 - src1);
    close |= (std::isfinite(actual_error) and (actual_error <= allowed_error));
    return close;
  }
  float atol, rtol;
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kLogicalAnd, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src0, Src src1) const { return static_cast<Dst>(src0 && src1); }
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kLogicalOr, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src0, Src src1) const { return static_cast<Dst>(src0 || src1); }
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kLogicalXor, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src0, Src src1) const {
    return static_cast<bool>(src0) != static_cast<bool>(src1);
  }
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kFmod, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src0, Src src1) const { return static_cast<Dst>(src0 % src1); }
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kFloorDiv, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src0, Src src1) const { return src0 / src1; }
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kTruncDiv, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src0, Src src1) const { return static_cast<Dst>(src0 / src1); }
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kFloorMod, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src0, Src src1) const {
    Src trunc_mod = src0 % src1;
    return (trunc_mod != static_cast<Src>(0))
                   && ((src1 < static_cast<Src>(0)) != (trunc_mod < static_cast<Src>(0)))
               ? trunc_mod + src1
               : trunc_mod;
  }
};

template<DeviceType device>
struct BinaryFunctor<device, BinaryOp::kFloorMod, uint8_t, uint8_t> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC uint8_t operator()(uint8_t src0, uint8_t src1) const { return src0 % src1; }
};

template<DeviceType device>
struct BinaryFunctor<device, BinaryOp::kFloorMod, uint32_t, uint32_t> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC uint32_t operator()(uint32_t src0, uint32_t src1) const { return src0 % src1; }
};

template<DeviceType device>
struct BinaryFunctor<device, BinaryOp::kFloorMod, uint64_t, uint64_t> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC uint64_t operator()(uint64_t src0, uint64_t src1) const { return src0 % src1; }
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kScalarBasePowerGrad, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) : scalar_operand(attr0.Value<Src>()) {}

  OF_DEVICE_FUNC Dst operator()(Src src0, Src src1) const {
    return scalar_operand * (pow(src0, scalar_operand - static_cast<Src>(1))) * src1;
  }
  Src scalar_operand;
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kScalarExpPowerGrad, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) : scalar_operand(attr0.Value<Src>()) {}

  OF_DEVICE_FUNC Dst operator()(Src src0, Src src1) const {
    return (pow(scalar_operand, src0)) * log(scalar_operand) * src1;
  }
  Src scalar_operand;
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kIdentityBackwardWithDyX, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src dy, Src x) const { return static_cast<Dst>(dy); }
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kEluBackwardWithDyX, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) : alpha(attr0.Value<double>()) {}

  OF_DEVICE_FUNC Dst operator()(Src dy, Src x) const {
    return (x > static_cast<Src>(0)) ? static_cast<Dst>(dy)
                                     : static_cast<Dst>(dy * alpha * (exp(x)));
  }
  const Src alpha;
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kCeluBackwardWithDyY, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1)
      : inv_alpha(1.0f / attr0.Value<double>()) {}

  OF_DEVICE_FUNC Dst operator()(Src dy, Src y) const {
    return static_cast<Dst>((y > static_cast<Src>(0))
                                ? dy
                                : dy * static_cast<Src>(y * inv_alpha + static_cast<Src>(1)));
  }
  const Src inv_alpha;
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kHardswishBackwardWithDyX, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src dy, Src x) const {
    if (x <= static_cast<Src>(-3)) {
      return static_cast<Dst>(0);
    } else if (x >= static_cast<Src>(3)) {
      return static_cast<Dst>(dy);
    } else {
      return static_cast<Dst>(((x / static_cast<Src>(3)) + static_cast<Src>(0.5)) * dy);
    }
  }
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kHardsigmoidBackwardWithDyX, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src dy, Src x) const {
    return static_cast<Dst>((x <= static_cast<Src>(-3) || x >= static_cast<Src>(3))
                                ? static_cast<Src>(0)
                                : dy / static_cast<Src>(6));
  }
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kHardshrinkBackwardWithDyY, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src dy, Src y) const {
    return static_cast<Dst>(y == static_cast<Src>(0) ? 0 : dy);
  }
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kHardtanhBackwardWithDyY, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1)
      : min_val(attr0.Value<float>()), max_val(attr1.Value<float>()) {}

  OF_DEVICE_FUNC Dst operator()(Src dy, Src y) const {
    return static_cast<Dst>((y == min_val || y == max_val) ? static_cast<Src>(0) : dy);
  }

  const Src min_val;
  const Src max_val;
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kLeakyReluBackwardWithDyX, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) : alpha(attr0.Value<float>()) {}

  OF_DEVICE_FUNC Dst operator()(Src dy, Src x) const {
    return static_cast<Dst>((x > static_cast<Src>(0)) ? dy : dy * alpha);
  }
  const Src alpha;
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kMishBackwardWithDyX, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src dy, Src x) const {
    Src sp = log(static_cast<Src>(1) + exp(x));
    Src grad_sp = static_cast<Src>(1) - exp(-sp);
    Src tsp = (exp(sp) - exp(-sp)) / (exp(sp) + exp(-sp));
    Src grad_tsp = (static_cast<Src>(1) - tsp * tsp) * grad_sp;
    return static_cast<Dst>(dy * (x * grad_tsp + tsp));
  }
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kReluBackwardWithDyY, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src dy, Src y) const {
    return static_cast<Dst>((y <= static_cast<Src>(0.0)) ? static_cast<Src>(0.0) : dy);
  }
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kReluBackwardWithDyX, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src dy, Src x) const {
    return static_cast<Dst>((x <= static_cast<Src>(0.0)) ? static_cast<Src>(0.0) : dy);
  }
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kSeluBackwardWithDyX, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src dy, Src x) const {
    return static_cast<Dst>((x > static_cast<Src>(0)) ? scale * dy : dy * scale * alpha * (exp(x)));
  }
  const Src scale = 1.0507009873554804934193349852946;
  const Src alpha = 1.6732632423543772848170429916717;
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kSiluBackwardWithDyX, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src dy, Src x) const {
    Src sig = static_cast<Src>(1) / (static_cast<Src>(1) + exp(-x));
    return static_cast<Dst>(dy * (sig * (static_cast<Src>(1) + x * (static_cast<Src>(1) - sig))));
  }
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kSoftsignBackwardWithDyX, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src dy, Src x) const {
    Src val = (static_cast<Src>(1) + abs(x));
    return static_cast<Dst>(dy / (val * val));
  }
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kSoftplusBackwardWithDyX, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1)
      : beta(attr0.Value<double>()), threshold(attr1.Value<double>()) {}

  OF_DEVICE_FUNC Dst operator()(Src dy, Src x) const {
    Src z = exp(x * beta);
    return static_cast<Dst>((x * beta) > threshold ? dy : dy * z / (z + static_cast<Src>(1.0)));
  }
  const Src beta;
  const Src threshold;
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kSoftshrinkBackwardWithDyY, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) : alpha(attr0.Value<double>()) {}

  OF_DEVICE_FUNC Dst operator()(Src dy, Src y) const {
    return static_cast<Dst>(y == static_cast<Src>(0) ? 0 : dy);
  }
  const Src alpha;
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kThresholdBackwardWithDyX, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) : threshold(attr0.Value<double>()) {}

  OF_DEVICE_FUNC Dst operator()(Src dy, Src x) const {
    return static_cast<Dst>((x <= threshold) ? 0 : dy);
  }
  const Src threshold;
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kAbsBackwardWithDyX, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src dy, Src x) const {
    const Src zero = static_cast<Src>(0.0);
    if (x == zero) {
      return zero;
    } else if (x < zero) {
      return -dy;
    } else {
      return dy;
    }
  }
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kAcosBackwardWithDyX, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}
  OF_DEVICE_FUNC Dst operator()(Src dy, Src x) const {
    return dy * -rsqrt(static_cast<Src>(1.0) - x * x);
  }
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kAcoshBackwardWithDyX, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}
  OF_DEVICE_FUNC Dst operator()(Src dy, Src x) const {
    return dy * rsqrt(x * x - static_cast<Src>(1.0));
  }
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kAsinBackwardWithDyX, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}
  OF_DEVICE_FUNC Dst operator()(Src dy, Src x) const {
    return dy * rsqrt(static_cast<Src>(1.0) - x * x);
  }
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kAsinhBackwardWithDyX, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}
  OF_DEVICE_FUNC Dst operator()(Src dy, Src x) const {
    return dy * rsqrt(static_cast<Src>(1.0) + x * x);
  }
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kAtanBackwardWithDyX, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}
  OF_DEVICE_FUNC Dst operator()(Src dy, Src x) const {
    const Src one = static_cast<Src>(1.0);
    return dy * (one / (one + x * x));
  }
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kAtanhBackwardWithDyX, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}
  OF_DEVICE_FUNC Dst operator()(Src dy, Src x) const {
    const Src one = static_cast<Src>(1.0);
    return dy * (one / (one - x * x));
  }
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kCosBackwardWithDyX, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}
  OF_DEVICE_FUNC Dst operator()(Src dy, Src x) const { return dy * (-sin(x)); }
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kCoshBackwardWithDyX, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}
  OF_DEVICE_FUNC Dst operator()(Src dy, Src x) const { return dy * sinh(x); }
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kErfBackwardWithDyX, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}
  OF_DEVICE_FUNC Dst operator()(Src dy, Src x) const {
    return dy * static_cast<Src>(2.0) * rsqrt(static_cast<Src>(M_PI)) * exp(-x * x);
  }
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kErfcBackwardWithDyX, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}
  OF_DEVICE_FUNC Dst operator()(Src dy, Src x) const {
    return dy * -static_cast<Src>(2.0) * rsqrt(static_cast<Src>(M_PI)) * exp(-x * x);
  }
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kExpBackwardWithDyX, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}
  OF_DEVICE_FUNC Dst operator()(Src dy, Src x) const { return dy * exp(x); }
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kExp2BackwardWithDyX, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}
  OF_DEVICE_FUNC Dst operator()(Src dy, Src x) const {
    return dy * exp2(x) * log(static_cast<Src>(2.0));
  }
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kExpm1BackwardWithDyX, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}
  OF_DEVICE_FUNC Dst operator()(Src dy, Src x) const { return dy * exp(x); }
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kLgammaBackwardWithDyX, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}
  OF_DEVICE_FUNC Dst operator()(Src dy, Src x) const {
    // TODO(chengcheng): return: dy * digamma(x)
    assert(false);
    return 0.0;
  }
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kLogBackwardWithDyX, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}
  OF_DEVICE_FUNC Dst operator()(Src dy, Src x) const { return dy * (static_cast<Src>(1.0) / x); }
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kLog2BackwardWithDyX, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}
  OF_DEVICE_FUNC Dst operator()(Src dy, Src x) const {
    return dy * (static_cast<Src>(1.0) / (x * log(static_cast<Src>(2.0))));
  }
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kLog10BackwardWithDyX, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}
  OF_DEVICE_FUNC Dst operator()(Src dy, Src x) const {
    return dy * (static_cast<Src>(1.0) / (x * log(static_cast<Src>(10.0))));
  }
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kLog1pBackwardWithDyX, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}
  OF_DEVICE_FUNC Dst operator()(Src dy, Src x) const {
    return dy * (static_cast<Src>(1.0) / (x + static_cast<Src>(1.0)));
  }
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kLogSigmoidBackwardWithDyX, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}
  OF_DEVICE_FUNC Dst operator()(Src dy, Src x) const {
    return dy * (static_cast<Src>(1.0) / (exp(x) + static_cast<Src>(1.0)));
  }
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kReciprocalBackwardWithDyX, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}
  OF_DEVICE_FUNC Dst operator()(Src dy, Src x) const {
    return dy * (-static_cast<Src>(1.0) / (x * x));
  }
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kReciprocalNoNanBackwardWithDyX, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}
  OF_DEVICE_FUNC Dst operator()(Src dy, Src x) const {
    if (abs(x) <= static_cast<Src>(0.0)) { return static_cast<Dst>(0.0); }
    return dy * (-static_cast<Src>(1.0) / (x * x));
  }
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kRsqrtBackwardWithDyX, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}
  OF_DEVICE_FUNC Dst operator()(Src dy, Src x) const {
    return dy * (static_cast<Src>(-1.0) / (static_cast<Src>(2.0) * sqrt(x * x * x)));
  }
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kSigmoidBackwardWithDyY, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}
  OF_DEVICE_FUNC Dst operator()(Src dy, Src y) const { return dy * (y * (1.0 - y)); }
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kSigmoidBackwardWithDyX, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}
  OF_DEVICE_FUNC Dst operator()(Src dy, Src x) const {
    using UnaryOp = oneflow::ep::primitive::UnaryOp;
    using UnaryFunctor = oneflow::ep::primitive::UnaryFunctor<device, UnaryOp::kSigmoid, Dst, Src>;
    auto uf = UnaryFunctor(0, 0);
    Src y = uf(x);
    return dy * (y * (static_cast<Src>(1.0) - y));
  }
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kSinBackwardWithDyX, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}
  OF_DEVICE_FUNC Dst operator()(Src dy, Src x) const { return dy * cos(x); }
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kSinhBackwardWithDyX, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}
  OF_DEVICE_FUNC Dst operator()(Src dy, Src x) const { return dy * cosh(x); }
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kSqrtBackwardWithDyX, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}
  OF_DEVICE_FUNC Dst operator()(Src dy, Src x) const {
    return dy * static_cast<Src>(0.5) / sqrt(x);
  }
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kSquareBackwardWithDyX, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}
  OF_DEVICE_FUNC Dst operator()(Src dy, Src x) const { return dy * static_cast<Src>(2.0) * x; }
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kTanBackwardWithDyX, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}
  OF_DEVICE_FUNC Dst operator()(Src dy, Src x) const {
    const Src cos_val = cos(x);
    return dy * (static_cast<Src>(1.0) / (cos_val * cos_val));
  }
};

}  // namespace broadcast_elementwise_binary
}  // namespace primitive
}  // namespace ep

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PRIMITIVE_COMMON_BINARY_FUNCTOR_H_
