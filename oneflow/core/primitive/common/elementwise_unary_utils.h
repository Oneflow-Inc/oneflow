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
#include "oneflow/core/primitive/include/elementwise_unary.h"
#include "oneflow/core/common/data_type.h"

namespace oneflow {

namespace primitive {

template<DeviceType device, UnaryOp unary_enum, typename Out, typename In>
struct UnaryFunctor;

template<DeviceType device, typename Out, typename In>
struct UnaryFunctor<device, UnaryOp::kRelu, Out, In> {
  OF_DEVICE_FUNC Out operator()(In src) const {
    const In zero_val = static_cast<In>(0.0);
    if (src > zero_val) {
      return static_cast<Out>(src);
    } else {
      return zero_val;
    }
  }
};

template<DeviceType device, typename Out, typename In>
struct UnaryFunctor<device, UnaryOp::kLogicalNot, Out, In> {
  OF_DEVICE_FUNC int8_t operator()(In src) const { return static_cast<int8_t>(!src); }
};

#define PRIMITIVE_SAME_DTYPE_UNARY_OP_SEQ OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kRelu, UnaryOp::kRelu)

#define PRIMITIVE_OUT_INT8_DTYPE_UNARY_OP_SEQ \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kLogicalNot, UnaryOp::kLogicalNot)

}  // namespace primitive
}  // namespace oneflow
