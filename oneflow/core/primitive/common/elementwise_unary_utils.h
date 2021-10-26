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

template<DeviceType device, UnaryOp unary_enum, typename T>
struct UnaryFunctor;

template<DeviceType device, typename T>
struct UnaryFunctor<device, UnaryOp::kRelu, T> {
  OF_DEVICE_FUNC T operator()(T src) const {
    const T zero_val = static_cast<T>(0.0);
    if (src > zero_val) {
      return src;
    } else {
      return zero_val;
    }
  }
};

#define PRIMITIVE_UNARY_OP_SEQ OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kRelu, UnaryOp::kRelu)

}  // namespace primitive
}  // namespace oneflow
