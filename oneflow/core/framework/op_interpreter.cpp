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
#include "oneflow/core/framework/op_interpreter.h"

namespace oneflow {
namespace one {

void OpExprInterpreter::ResetSelfState() { self_state_.reset(new OpExprInterpState); }

void NormalInterpreter::Apply(const OpExpr* op_expr, const TensorList& inputs, TensorList& outputs,
                              const OpExprInterpState* state) {
  ResetSelfState();
  if (op_expr->type() == "UserOp") {
    Apply_(dynamic_cast<const UserOpExpr*>(op_expr), inputs, outputs, state);
  } else if (op_expr->type() == "FunctionOp") {
    Apply_(dynamic_cast<const FunctionOpExpr*>(op_expr), inputs, outputs, state);
  } else {
    LOG(FATAL) << "The op type " << op_expr->type()
               << " is not supported in LazyInterpreter::Apply currently.";
  }
}

void LazyInterpreter::Apply_(const UserOpExpr* op_expr, const TensorList& inputs,
                             TensorList& outputs, const OpExprInterpState* state) {}

void EagerInterpreter::Apply_(const UserOpExpr* op_expr, const TensorList& inputs,
                              TensorList& outputs, const OpExprInterpState* state) {}

void LazyInterpreter::Apply_(const FunctionOpExpr* op_expr, const TensorList& inputs,
                             TensorList& outputs, const OpExprInterpState* state) {
  // TODO(hjchen2)
}

void EagerInterpreter::Apply_(const FunctionOpExpr* op_expr, const TensorList& inputs,
                              TensorList& outputs, const OpExprInterpState* state) {
  // TODO(hjchen2)
}

void AutogradInterpreter::Apply(const OpExpr* op_expr, const TensorList& inputs,
                                TensorList& outputs, const OpExprInterpState* state) {
  // TODO(hjchen2)
}

}  // namespace one
}  // namespace oneflow
