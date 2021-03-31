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
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/framework/op_arg_util.h"
#include "oneflow/core/framework/scope_util.h"
#include "oneflow/core/framework/session_util.h"
#include "oneflow/core/framework/symbol_storage_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_name_scope.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/eager/foreign_boxing_util.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {
namespace one {

static Maybe<void> NaiveInterpret(const BuiltinOpExpr& op_expr, const TensorTuple& inputs,
                                  TensorTuple* outputs) {
  // TODO()
  UNIMPLEMENTED();
}

Maybe<void> EagerMirroredInterpreter::ApplyImpl(const UserOpExpr& op_expr,
                                                const TensorTuple& inputs,
                                                TensorTuple* outputs) const {
  return NaiveInterpret(op_expr, inputs, outputs);
}

Maybe<void> EagerMirroredInterpreter::ApplyImpl(const VariableOpExpr& op_expr,
                                                const TensorTuple& inputs,
                                                TensorTuple* outputs) const {
  CHECK_EQ_OR_RETURN(inputs.size(), 0);
  CHECK_EQ_OR_RETURN(outputs->size(), 1);
  return NaiveInterpret(op_expr, inputs, outputs);
}

static Maybe<void> BuildAndRunMirroredCastInstruction(const BuiltinOpExpr& op_expr,
                                                      const TensorTuple& inputs,
                                                      TensorTuple* outputs) {
  // TODO()
  UNIMPLEMENTED();
}

Maybe<void> EagerMirroredInterpreter::ApplyImpl(const CastToMirroredOpExpr& op_expr,
                                                const TensorTuple& inputs,
                                                TensorTuple* outputs) const {
  return BuildAndRunMirroredCastInstruction(op_expr, inputs, outputs);
}

Maybe<void> EagerMirroredInterpreter::ApplyImpl(const CastFromMirroredOpExpr& op_expr,
                                                const TensorTuple& inputs,
                                                TensorTuple* outputs) const {
  return BuildAndRunMirroredCastInstruction(op_expr, inputs, outputs);
}

static Maybe<void> BuildAndRunDistributeSplitOrCloneInstruction(const BuiltinOpExpr& op_expr,
                                                                const TensorTuple& inputs,
                                                                TensorTuple* outputs) {
  // TODO()
  UNIMPLEMENTED();
}

Maybe<void> EagerMirroredInterpreter::ApplyImpl(const DistributeSplitOpExpr& op_expr,
                                                const TensorTuple& inputs,
                                                TensorTuple* outputs) const {
  return BuildAndRunDistributeSplitOrCloneInstruction(op_expr, inputs, outputs);
}

Maybe<void> EagerMirroredInterpreter::ApplyImpl(const DistributeCloneOpExpr& op_expr,
                                                const TensorTuple& inputs,
                                                TensorTuple* outputs) const {
  return BuildAndRunDistributeSplitOrCloneInstruction(op_expr, inputs, outputs);
}

static Maybe<void> BuildAndRunDistributeConcatAndAddInstruction(const BuiltinOpExpr& op_expr,
                                                                const TensorTuple& inputs,
                                                                TensorTuple* outputs) {
  // TODO()
  UNIMPLEMENTED();
}

Maybe<void> EagerMirroredInterpreter::ApplyImpl(const DistributeConcatOpExpr& op_expr,
                                                const TensorTuple& inputs,
                                                TensorTuple* outputs) const {
  return BuildAndRunDistributeConcatAndAddInstruction(op_expr, inputs, outputs);
}

Maybe<void> EagerMirroredInterpreter::ApplyImpl(const DistributeAddOpExpr& op_expr,
                                                const TensorTuple& inputs,
                                                TensorTuple* outputs) const {
  return BuildAndRunDistributeConcatAndAddInstruction(op_expr, inputs, outputs);
}

}  // namespace one
}  // namespace oneflow
