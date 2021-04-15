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
#ifndef ONEFLOW_CORE_FRAMEWORK_OP_DISPATCH_H_
#define ONEFLOW_CORE_FRAMEWORK_OP_DISPATCH_H_

#include "oneflow/core/framework/attr_value_map.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"

namespace oneflow {
namespace one {

template<typename T>
Maybe<T> Dispatch(const OpExpr& op_expr, const TensorTuple& inputs, const AttrValueMap& attrs);

template<typename T>
Maybe<T> Dispatch(const OpExpr& op_expr, const TensorTuple& inputs) {
  return Dispatch<T>(op_expr, inputs, AttrValueMap{});
}

}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_OP_DISPATCH_H_
