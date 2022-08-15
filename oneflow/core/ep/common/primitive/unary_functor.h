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
#ifndef ONEFLOW_CORE_EP_COMMON_PRIMITIVE_UNARY_FUNCTOR_H_
#define ONEFLOW_CORE_EP_COMMON_PRIMITIVE_UNARY_FUNCTOR_H_

#include "oneflow/core/ep/include/primitive/unary_op.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/scalar.h"

namespace oneflow {

namespace ep {
namespace primitive {

template<DeviceType device, UnaryOp unary_op, typename Dst, typename Src>
struct UnaryFunctor;

template<DeviceType device, typename Dst, typename Src>
struct UnaryFunctor<device, UnaryOp::kIdentity, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const { return static_cast<Dst>(src); }
};

template<DeviceType device, typename Dst, typename Src>
struct UnaryFunctor<device, UnaryOp::kElu, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) : alpha(attr0.Value<double>()) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const {
    return static_cast<Dst>(
        (src > static_cast<Src>(0.0)) ? src : alpha * (exp(src) - static_cast<Src>(1)));
  }
  const Src alpha;
};

template<DeviceType device, typename Dst, typename Src>
struct UnaryFunctor<device, UnaryOp::kCelu, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1)
      : alpha(attr0.Value<double>()), inv_alpha(1.0f / attr0.Value<double>()) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const {
    return static_cast<Dst>(
        (src > static_cast<Src>(0.0)) ? src : alpha * (exp(src * inv_alpha) - static_cast<Src>(1)));
  }
  const Src alpha;
  const Src inv_alpha;
};

template<DeviceType device, typename Dst, typename Src>
struct UnaryFunctor<device, UnaryOp::kHardSwish, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const {
    if (src <= static_cast<Src>(-3)) {
      return static_cast<Dst>(0);
    } else if (src >= static_cast<Src>(3)) {
      return static_cast<Dst>(src);
    } else {
      return static_cast<Dst>((src * (src + static_cast<Src>(3))) / static_cast<Src>(6));
    }
  }
};

template<DeviceType device, typename Dst, typename Src>
struct UnaryFunctor<device, UnaryOp::kHardSigmoid, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const {
    if (src <= static_cast<Src>(-3)) {
      return static_cast<Dst>(0);
    } else if (src >= static_cast<Src>(3)) {
      return static_cast<Dst>(1);
    } else {
      return static_cast<Dst>(src / static_cast<Src>(6) + static_cast<Src>(0.5));
    }
  }
};

template<DeviceType device, typename Dst, typename Src>
struct UnaryFunctor<device, UnaryOp::kHardShrink, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) : lambd(attr0.Value<double>()) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const {
    return (src <= lambd && src >= -lambd) ? static_cast<Dst>(0) : static_cast<Dst>(src);
  }

  const Src lambd;
};

template<DeviceType device, typename Dst, typename Src>
struct UnaryFunctor<device, UnaryOp::kHardTanh, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1)
      : min_val(attr0.Value<double>()), max_val(attr1.Value<double>()) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const {
    if (src <= min_val) {
      return static_cast<Dst>(min_val);
    } else if (src >= max_val) {
      return static_cast<Dst>(max_val);
    } else {
      return static_cast<Dst>(src);
    }
  }

  const Src min_val;
  const Src max_val;
};

template<DeviceType device, typename Dst, typename Src>
struct UnaryFunctor<device, UnaryOp::kLeakyRelu, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) : alpha(attr0.Value<float>()) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const {
    return static_cast<Dst>((src > static_cast<Src>(0.0)) ? src : alpha * src);
  }
  const Src alpha;
};

template<DeviceType device, typename Dst, typename Src>
struct UnaryFunctor<device, UnaryOp::kMish, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const {
    Src soft_plus_val = log(static_cast<Src>(1) + exp(src));
    Src exp_val = exp(soft_plus_val);
    Src neg_exp_val = exp(-soft_plus_val);
    Src tanh_val = (exp_val - neg_exp_val) / (exp_val + neg_exp_val);
    return static_cast<Dst>(src * tanh_val);
  }
};

template<DeviceType device, typename Dst, typename Src>
struct UnaryFunctor<device, UnaryOp::kRelu, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const {
    const Src zero_val = static_cast<Src>(0.0);
    if (src <= zero_val) {
      return static_cast<Dst>(zero_val);
    } else {
      return static_cast<Dst>(src);
    }
  }
};

template<DeviceType device, typename Dst, typename Src>
struct UnaryFunctor<device, UnaryOp::kSilu, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const {
    return static_cast<Dst>(src / (static_cast<Src>(1) + exp(-src)));
  }
};

template<DeviceType device, typename Dst, typename Src>
struct UnaryFunctor<device, UnaryOp::kSelu, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const {
    return static_cast<Dst>((src > static_cast<Src>(0.0))
                                ? src * scale
                                : scale * alpha * (exp(src) - static_cast<Src>(1)));
  }
  const Src scale = 1.0507009873554804934193349852946;
  const Src alpha = 1.6732632423543772848170429916717;
};

template<DeviceType device, typename Dst, typename Src>
struct UnaryFunctor<device, UnaryOp::kSoftSign, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const {
    return static_cast<Dst>(src / (static_cast<Src>(1) + abs(src)));
  }
};

template<DeviceType device, typename Dst, typename Src>
struct UnaryFunctor<device, UnaryOp::kSoftPlus, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1)
      : beta(attr0.Value<double>()), threshold(attr1.Value<double>()) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const {
    return static_cast<Dst>(
        (src * beta) > threshold ? src : log(static_cast<Src>(1.0) + exp(src * beta)) / beta);
  }

  const Src beta;
  const Src threshold;
};

template<DeviceType device, typename Dst, typename Src>
struct UnaryFunctor<device, UnaryOp::kSoftShrink, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) : alpha(attr0.Value<double>()) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const {
    if (src <= alpha && src >= -alpha) {
      return static_cast<Dst>(0);
    } else if (src > alpha) {
      return static_cast<Dst>(src - alpha);
    } else {
      return static_cast<Dst>(src + alpha);
    }
  }
  const Src alpha;
};

template<DeviceType device, typename Dst, typename Src>
struct UnaryFunctor<device, UnaryOp::kThreshold, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1)
      : threshold(attr0.Value<double>()), value(attr1.Value<double>()) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const {
    return static_cast<Dst>((src <= threshold) ? value : src);
  }
  const Src threshold;
  const Src value;
};

template<DeviceType device, typename Dst, typename Src>
struct UnaryFunctor<device, UnaryOp::kLogicalNot, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const { return static_cast<Dst>(!src); }
};

template<DeviceType device, typename Src>
struct UnaryFunctor<device, UnaryOp::kIsInf, bool, Src> {
  UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC bool operator()(Src src) const { return false; }
};

template<DeviceType device, typename Src>
struct UnaryFunctor<device, UnaryOp::kIsNan, bool, Src> {
  UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC bool operator()(Src src) const { return false; }
};

template<DeviceType device, typename Src>
struct UnaryFunctor<device, UnaryOp::kIsFinite, bool, Src> {
  UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC bool operator()(Src src) const { return true; }
};

}  // namespace primitive
}  // namespace ep
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EP_COMMON_PRIMITIVE_UNARY_FUNCTOR_H_
