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
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/scalar.h"

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
struct BinaryFunctor<device, BinaryOp::kEluBackwardWithDyX, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) : alpha(attr0.Value<float>()) {}

  OF_DEVICE_FUNC Dst operator()(Src src0, Src src1) const {
    return (src0 > static_cast<Src>(0)) ? static_cast<Dst>(src1)
                                        : static_cast<Dst>(src1 * alpha * (exp(src0)));
  }
  const Src alpha;
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kCeluBackwardWithDyX, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1)
      : inv_alpha(1.0f / attr0.Value<double>()) {}

  OF_DEVICE_FUNC Dst operator()(Src src0, Src src1) const {
    return static_cast<Dst>(
        (src0 > static_cast<Src>(0)) ? src1 : src1 * static_cast<Src>(exp(src0 * inv_alpha)));
  }
  const Src inv_alpha;
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kHardswishBackwardWithDyX, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src0, Src src1) const {
    if (src0 <= static_cast<Src>(-3)) {
      return static_cast<Dst>(0);
    } else if (src0 >= static_cast<Src>(3)) {
      return static_cast<Dst>(src1);
    } else {
      return static_cast<Dst>(((src0 / static_cast<Src>(3)) + static_cast<Src>(0.5)) * src1);
    }
  }
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kHardsigmoidBackwardWithDyX, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src0, Src src1) const {
    return static_cast<Dst>((src0 > static_cast<Src>(-3) && src0 < static_cast<Src>(3))
                                ? src1 / static_cast<Src>(6)
                                : static_cast<Src>(0));
  }
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kHardshrinkBackwardWithDyY, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src0, Src src1) const {
    return static_cast<Dst>(src0 == static_cast<Src>(0) ? 0 : src1);
  }
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kHardtanhBackwardWithDyY, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1)
      : min_val(attr0.Value<float>()), max_val(attr1.Value<float>()) {}

  OF_DEVICE_FUNC Dst operator()(Src src0, Src src1) const {
    return static_cast<Dst>((src0 != min_val && src0 != max_val) ? src1 : static_cast<Src>(0));
  }

  const Src min_val;
  const Src max_val;
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kLeakyReluBackwardWithDyX, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) : alpha(attr0.Value<float>()) {}

  OF_DEVICE_FUNC Dst operator()(Src src0, Src src1) const {
    return static_cast<Dst>((src0 > static_cast<Src>(0)) ? src1 : src1 * alpha);
  }
  const Src alpha;
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kMishBackwardWithDyX, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src0, Src src1) const {
    Src sp = log(static_cast<Src>(1) + exp(src0));
    Src grad_sp = static_cast<Src>(1) - exp(-sp);
    Src tsp = (exp(sp) - exp(-sp)) / (exp(sp) + exp(-sp));
    Src grad_tsp = (static_cast<Src>(1) - tsp * tsp) * grad_sp;
    return static_cast<Dst>(src1 * (src0 * grad_tsp + tsp));
  }
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kReluBackwardWithDyY, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src0, Src src1) const {
    return static_cast<Dst>((src0 > static_cast<Src>(0)) * src1);
  }
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kSeluBackwardWithDyX, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src0, Src src1) const {
    return static_cast<Dst>((src0 > static_cast<Src>(0)) ? scale * src1
                                                         : src1 * scale * alpha * (exp(src0)));
  }
  const Src scale = 1.0507009873554804934193349852946;
  const Src alpha = 1.6732632423543772848170429916717;
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kSiluBackwardWithDyX, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src0, Src src1) const {
    Src sig = static_cast<Src>(1) / (static_cast<Src>(1) + exp(-src0));
    return static_cast<Dst>(src1
                            * (sig * (static_cast<Src>(1) + src0 * (static_cast<Src>(1) - sig))));
  }
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kSoftsignBackwardWithDyX, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src0, Src src1) const {
    Src val = (static_cast<Src>(1) + abs(src0));
    return static_cast<Dst>(src1 / (val * val));
  }
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kSoftplusBackwardWithDyX, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1)
      : beta(attr0.Value<double>()), threshold(attr1.Value<double>()) {}

  OF_DEVICE_FUNC Dst operator()(Src src0, Src src1) const {
    Src z = exp(src0 * beta);
    return static_cast<Dst>((src0 * beta) > threshold ? src1
                                                      : src1 * z / (z + static_cast<Src>(1.0)));
  }
  const Src beta;
  const Src threshold;
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kSoftshrinkBackwardWithDyY, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) : alpha(attr0.Value<double>()) {}

  OF_DEVICE_FUNC Dst operator()(Src src0, Src src1) const {
    return static_cast<Dst>(src0 == static_cast<Src>(0) ? 0 : src1);
  }
  const Src alpha;
};

template<DeviceType device, typename Src, typename Dst>
struct BinaryFunctor<device, BinaryOp::kThresholdBackwardWithDyX, Src, Dst> {
  OF_DEVICE_FUNC BinaryFunctor(Scalar attr0, Scalar attr1) : threshold(attr0.Value<double>()) {}

  OF_DEVICE_FUNC Dst operator()(Src src0, Src src1) const {
    return static_cast<Dst>((src0 > threshold) ? src1 : 0);
  }
  const Src threshold;
};

}  // namespace broadcast_elementwise_binary
}  // namespace primitive
}  // namespace ep

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PRIMITIVE_COMMON_BINARY_FUNCTOR_H_
