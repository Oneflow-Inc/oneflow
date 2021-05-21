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

#include "oneflow/core/functional/functional.h"

#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_expr_helper.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"

namespace oneflow {
namespace one {
namespace functional {

Maybe<one::Tensor> Add(const TensorTuple& inputs, const AttrMap& attrs) {
  static thread_local const auto& op = CHECK_JUST(op_expr_helper::AddOp());
  return OpInterpUtil::Dispatch<Tensor>(*op, inputs, attrs);
}

Maybe<one::Tensor> AddScalar(const TensorTuple& inputs, const AttrMap& attrs) {
  static thread_local const auto& op = CHECK_JUST(op_expr_helper::ScalarAddOp<float>(0));
  return OpInterpUtil::Dispatch<Tensor>(*op, inputs, attrs);
}

}  // namespace functional
}  // namespace one
}  // namespace oneflow
