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
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/framework/op_arg_util.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/api/python/job_build/job_build_and_infer.h"

namespace oneflow {
namespace one {

static std::shared_ptr<cfg::OpAttribute> AddOpAndInferOpAttribute(
    const BuiltinOpExpr* op_expr, const OpExprInterpContext* ctx,
    const std::unordered_map<std::string, std::string>& ibn2tensor_names = {}) {
  OperatorConf op_conf;
  op_expr->BuildOpConf(&op_conf);
  for (const auto& it : ibn2tensor_names) {
    ReplaceInputLbnInOpCustomizedConf(&op_conf, it.first, it.second);
  }
  int64_t symbol_id = ctx->scope->symbol_id().GetOrThrow();
  op_conf.set_scope_symbol_id(symbol_id);
  if (!op_conf.has_device_tag()) {
    op_conf.set_device_tag(ctx->scope->device_parallel_desc_symbol()->device_tag());
  }
  OpAttribute op_attribute = [&]() {
    auto infer_ctx = GetCurInferCtx().GetOrThrow();
    if (ctx->is_mirrored_strategy_enabled) {
      return infer_ctx->AddAndInferMirroredOp(op_conf).GetOrThrow();
    } else {
      return infer_ctx->AddAndInferConsistentOp(op_conf).GetOrThrow();
    }
  }();
  return std::make_shared<cfg::OpAttribute>(op_attribute);
}

void OpExprInterpreter::ResetSelfState() { self_state_.reset(new OpExprInterpState); }

void LazyInterpreter::Apply(const OpExpr* op_expr, const TensorList& inputs, TensorList& outputs,
                            const OpExprInterpState* state) {
  ResetSelfState();

#define APPLY_IF(op_type)                                             \
  if (const auto* op = dynamic_cast<const op_type##Expr*>(op_expr)) { \
    return Apply_(op, inputs, outputs, state);                        \
  }

  APPLY_IF(FunctionOp);
  APPLY_IF(BuiltinOp);
#undef APPLY_IF

  LOG(FATAL) << "The type " << op_expr->type()
             << " has not been supported in LazyInterpreter::Apply.";
}

void LazyInterpreter::Apply_(const BuiltinOpExpr* op_expr, const TensorList& inputs,
                             TensorList& outputs, const OpExprInterpState* state) {
  CHECK_EQ(inputs.size(), op_expr->input_num());
  std::unordered_map<std::string, std::string> ibn2tensor_names;
  for (int i = 0; i < inputs.size(); ++i) {
    const std::string& ibn = op_expr->indexed_ibns().at(i);
    ibn2tensor_names[ibn] = TensorNameScope::Global()->Lookup(inputs[i]);
  }
  auto op_attribute = AddOpAndInferOpAttribute(op_expr, context(), ibn2tensor_names);

  // Check outputs num and setup output tensor properties.
  CHECK_EQ(outputs.size(), op_expr->output_num());
  for (int i = 0; i < op_expr->output_num(); ++i) {
    // TODO
    const std::string& obn = op_expr->op_name() + "_" + op_expr->indexed_obns().at(i);
    TensorNameScope::Global()->Record(outputs[i], obn);
  }
}

void LazyInterpreter::Apply_(const FunctionOpExpr* op_expr, const TensorList& inputs,
                             TensorList& outputs, const OpExprInterpState* state) {
  // TODO
}

void EagerInterpreter::Apply(const OpExpr* op_expr, const TensorList& inputs, TensorList& outputs,
                             const OpExprInterpState* state) {
  ResetSelfState();

#define APPLY_IF(op_type)                                             \
  if (const auto* op = dynamic_cast<const op_type##Expr*>(op_expr)) { \
    return Apply_(op, inputs, outputs, state);                        \
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

  LOG(FATAL) << "The type " << op_expr->type()
             << " has not been supported in EagerInterpreter::Apply.";
}

void EagerInterpreter::Apply_(const UserOpExpr* op_expr, const TensorList& inputs,
                              TensorList& outputs, const OpExprInterpState* state) {
  auto op_attribute = AddOpAndInferOpAttribute(op_expr, context());
  auto parallel_conf = std::make_shared<cfg::ParallelConf>(
      context()->scope->device_parallel_desc_symbol()->parallel_conf());
  auto BuildInstruction = [&](const std::shared_ptr<InstructionsBuilder>& builder) {
    // TODO(hjchen2) Complete bn2blob_object.
    std::shared_ptr<HashMap<std::string, std::shared_ptr<compatible_py::BlobObject>>>
        bn2blob_object;
    builder->NoBoxingStatelessCall(op_attribute, parallel_conf, bn2blob_object);
  };
  void(LogicalRun(BuildInstruction).GetOrThrow());
}

void EagerInterpreter::Apply_(const VariableOpExpr* op_expr, const TensorList& inputs,
                              TensorList& outputs, const OpExprInterpState* state) {
  // TODO(hjchen2)
}

static std::function<void(const std::shared_ptr<InstructionsBuilder>& builder)>
BuildMirroredCastInstruction(const BuiltinOpExpr* op_expr, const OpExprInterpContext* ctx) {
  auto op_attribute = AddOpAndInferOpAttribute(op_expr, ctx);
  auto BuildInstruction = [&](const std::shared_ptr<InstructionsBuilder>& builder) {
    std::shared_ptr<HashMap<std::string, std::shared_ptr<compatible_py::BlobObject>>>
        bn2blob_object;
    const auto& in_blob_object = (*bn2blob_object)["in"];
    const auto& parallel_desc_symbol = in_blob_object->parallel_desc_symbol();
    OpAttribute proto_op_attribute;
    op_attribute->ToProto(&proto_op_attribute);
    const auto& op_arg_parallel_attr =
        compatible_py::GetOpArgParallelAttribute(parallel_desc_symbol, proto_op_attribute, "out")
            .GetOrThrow();
    auto out_blob_object = builder->MakeReferenceBlobObject(
        in_blob_object,
        std::make_shared<compatible_py::OpArgParallelAttribute>(op_arg_parallel_attr));
    // (*bn2blob_object)["out"] = out_blob_object;
  };
  return BuildInstruction;
}

void EagerInterpreter::Apply_(const CastToMirroredOpExpr* op_expr, const TensorList& inputs,
                              TensorList& outputs, const OpExprInterpState* state) {
  auto BuildInstruction = BuildMirroredCastInstruction(op_expr, context());
  LogicalRun(BuildInstruction).GetOrThrow();
}

void EagerInterpreter::Apply_(const CastFromMirroredOpExpr* op_expr, const TensorList& inputs,
                              TensorList& outputs, const OpExprInterpState* state) {
  auto BuildInstruction = BuildMirroredCastInstruction(op_expr, context());
  LogicalRun(BuildInstruction).GetOrThrow();
}

static std::function<void(const std::shared_ptr<InstructionsBuilder>& builder)>
BuildDistributeSplitOrCloneInstruction(const BuiltinOpExpr* op_expr,
                                       const OpExprInterpContext* ctx) {
  auto op_attribute = AddOpAndInferOpAttribute(op_expr, ctx);
  auto parallel_conf = std::make_shared<cfg::ParallelConf>(
      ctx->scope->device_parallel_desc_symbol()->parallel_conf());
  auto BuildInstruction = [&](const std::shared_ptr<InstructionsBuilder>& builder) {

  };
}

void EagerInterpreter::Apply_(const DistributeSplitOpExpr* op_expr, const TensorList& inputs,
                              TensorList& outputs, const OpExprInterpState* state) {
  auto BuildInstruction = BuildDistributeSplitOrCloneInstruction(op_expr, context());
  LogicalRun(BuildInstruction).GetOrThrow();
}

void EagerInterpreter::Apply_(const DistributeCloneOpExpr* op_expr, const TensorList& inputs,
                              TensorList& outputs, const OpExprInterpState* state) {
  auto BuildInstruction = BuildDistributeSplitOrCloneInstruction(op_expr, context());
  LogicalRun(BuildInstruction).GetOrThrow();
}

static std::function<void(const std::shared_ptr<InstructionsBuilder>& builder)>
BuildDistributeConcatAndAddInstruction(const BuiltinOpExpr* op_expr,
                                       const OpExprInterpContext* ctx) {}

void EagerInterpreter::Apply_(const DistributeConcatOpExpr* op_expr, const TensorList& inputs,
                              TensorList& outputs, const OpExprInterpState* state) {
  auto BuildInstruction = BuildDistributeConcatAndAddInstruction(op_expr, context());
  LogicalRun(BuildInstruction).GetOrThrow();
}

void EagerInterpreter::Apply_(const DistributeAddOpExpr* op_expr, const TensorList& inputs,
                              TensorList& outputs, const OpExprInterpState* state) {
  auto BuildInstruction = BuildDistributeConcatAndAddInstruction(op_expr, context());
  LogicalRun(BuildInstruction).GetOrThrow();
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
