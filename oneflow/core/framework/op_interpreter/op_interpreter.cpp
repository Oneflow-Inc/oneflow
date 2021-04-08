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

Maybe<void> LazyInterpreter::Apply(const OpExpr& op_expr, const TensorTuple& inputs,
                                   TensorTuple* outputs) const {
#define APPLY_IF(op_type)                                              \
  if (const auto* op = dynamic_cast<const op_type##Expr*>(&op_expr)) { \
    return ApplyImpl(*op, inputs, outputs);                            \
  }

  APPLY_IF(FunctionOp);
  APPLY_IF(BuiltinOp);
#undef APPLY_IF

  OF_UNIMPLEMENTED() << "The type " << op_expr.type()
                     << " has not been supported in LazyInterpreter::Apply.";
}

Maybe<void> LazyInterpreter::ApplyImpl(const BuiltinOpExpr& op_expr, const TensorTuple& inputs,
                                       TensorTuple* outputs) const {
  CHECK_EQ_OR_RETURN(inputs.size(), op_expr.input_num());
  const auto& scope = JUST(GetCurrentScope());
  auto op_conf = JUST(OpInterpUtil::GenBuiltinOpConf(op_expr));
  int64_t symbol_id = JUST(scope->symbol_id());
  op_conf->set_scope_symbol_id(symbol_id);
  if (!op_conf->has_device_tag()) {
    op_conf->set_device_tag(scope->device_parallel_desc_symbol()->device_tag());
  }
  for (int i = 0; i < inputs.size(); ++i) {
    const std::string& ibn = op_expr.indexed_ibns().at(i);
    const std::string& tensor_name = TensorNameScope::Global()->Lookup(inputs[i]);
    ReplaceInputLbnInOpCustomizedConf(op_conf.get(), ibn, tensor_name);
  }
  const auto& session = JUST(GetDefaultSession());
  bool is_mirrored_strategy_enabled = JUST(session->IsMirroredStrategyEnabled());
  const auto& op_attribute =
      JUST(OpInterpUtil::AddOpAndInferOpAttribute(*op_conf, is_mirrored_strategy_enabled));
  OpAttribute proto_op_attribute;
  op_attribute->ToProto(&proto_op_attribute);

  int64_t parallel_desc_sym_id = JUST(scope->GetParallelDescSymbolId(*op_conf));
  const std::shared_ptr<ParallelDesc>& blob_parallel_desc_sym =
      JUST(GetSymbol<cfg::ParallelConf, ParallelDesc>(parallel_desc_sym_id));

  // Check outputs num and setup output tensor properties.
  CHECK_EQ_OR_RETURN(outputs->size(), op_expr.output_num());
  for (int i = 0; i < op_expr.output_num(); ++i) {
    const std::string& obn = op_expr.indexed_obns().at(i);
    const auto& parallel_attr = JUST(
        compatible_py::GetOpArgParallelAttribute(blob_parallel_desc_sym, proto_op_attribute, obn));
    const auto& blob_attr = JUST(compatible_py::GetOpArgBlobAttribute(proto_op_attribute, obn));
    if (!(outputs->at(i).get())) {
      (*outputs)[i] = JUST(OpInterpUtil::BuildTensor(blob_attr, parallel_attr, /*is_lazy=*/true));
    } else {
      // TODO(hjchen2) Reset shape, dtype and so on.
      UNIMPLEMENTED();
    }
    TensorNameScope::Global()->Record(outputs->at(i), op_expr.op_name() + "/" + obn);
  }
  return Maybe<void>::Ok();
}

Maybe<void> LazyInterpreter::ApplyImpl(const FunctionOpExpr& op_expr, const TensorTuple& inputs,
                                       TensorTuple* outputs) const {
  // TODO(hjchen2)
  UNIMPLEMENTED();
  return Maybe<void>::Ok();
}

Maybe<void> EagerInterpreter::Apply(const OpExpr& op_expr, const TensorTuple& inputs,
                                    TensorTuple* outputs) const {
#define APPLY_IF(op_type)                                              \
  if (const auto* op = dynamic_cast<const op_type##Expr*>(&op_expr)) { \
    return ApplyImpl(*op, inputs, outputs);                            \
  }

  APPLY_IF(UserOp);
  APPLY_IF(VariableOp);
  APPLY_IF(CastToMirroredOp);
  APPLY_IF(CastFromMirroredOp);
  APPLY_IF(DistributeSplitOp);
  APPLY_IF(DistributeCloneOp);
  APPLY_IF(DistributeConcatOp);
  APPLY_IF(DistributeAddOp);
  APPLY_IF(FunctionOp);
#undef APPLY_IF

  OF_UNIMPLEMENTED() << "The type " << op_expr.type()
                     << " has not been supported in EagerInterpreter::Apply.";
}

Maybe<void> EagerInterpreter::ApplyImpl(const FunctionOpExpr& op_expr, const TensorTuple& inputs,
                                        TensorTuple* outputs) const {
  // TODO(hjchen2)
  UNIMPLEMENTED();
  return Maybe<void>::Ok();
}

Maybe<void> AutogradInterpreter::Apply(const OpExpr& op_expr, const TensorTuple& inputs,
                                       TensorTuple* outputs) const {
  // TODO(hjchen2)
  return internal_->Apply(op_expr, inputs, outputs);
}

}  // namespace one
}  // namespace oneflow
