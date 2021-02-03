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
#include "oneflow/core/framework/op_expr.h"

namespace oneflow {
namespace one {

TensorList UserOpExpr::evaluate(OpExprEvaluator* evaluator, const TensorList& inputs,
                                const OpExprEvalState* state) {
  // TODO(hjchen2)
  return TensorList{};
}

std::shared_ptr<OpExpr> UserOpExpr::GetBackwardOpExpr() {
  // TODO(hjchen2)
  return std::shared_ptr<OpExpr>(new UserOpExpr);
}

TensorList NormalEvaluator::apply(const OpExpr* fw_op_expr, const TensorList& inputs,
                                  const OpExprEvalState* state) {
  // TODO(hjchen2)
  return TensorList{};
}

TensorList AutogradEvaluator::apply(const OpExpr* fw_op_expr, const TensorList& inputs,
                                    const OpExprEvalState* state) {
  // TODO(hjchen2)
  return TensorList{};
}

}  // namespace one
}  // namespace oneflow
