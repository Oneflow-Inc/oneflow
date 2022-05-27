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
struct UnaryFunctor<device, UnaryOp::kRelu, Dst, Src> {
  UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const {
    const Src zero_val = static_cast<Src>(0.0);
    if (src > zero_val) {
      return static_cast<Dst>(src);
    } else {
      return static_cast<Dst>(zero_val);
    }
  }
};

template<DeviceType device, typename Dst, typename Src>
struct UnaryFunctor<device, UnaryOp::kLogicalNot, Dst, Src> {
  UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const { return static_cast<Dst>(!src); }
};

template<DeviceType device, typename Dst, typename Src>
struct UnaryFunctor<device, UnaryOp::kLeakyRelu, Dst, Src> {
  UnaryFunctor(Scalar attr0, Scalar attr1) : alpha(attr0.Value<float>()) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const {
    return (src > static_cast<Src>(0.0)) ? src : alpha * src;
  }
  const Src alpha;
};

}  // namespace primitive
}  // namespace ep
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EP_COMMON_PRIMITIVE_UNARY_FUNCTOR_H_
