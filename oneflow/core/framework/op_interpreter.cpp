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
#include "oneflow/core/framework/op_interpreter_util.h"
#include "oneflow/core/framework/op_expr_grad.h"
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
      auto t = JUST(OpInterpUtil::BuildTensor(blob_attr, parallel_attr, /*is_lazy=*/true));
      outputs->at(i).swap(t);
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

static Maybe<void> NaiveInterpret(const BuiltinOpExpr& op_expr, const TensorTuple& inputs,
                                  TensorTuple* outputs) {
  using namespace std::placeholders;
  const auto& scope = JUST(GetCurrentScope());
  const auto& op_attribute = JUST(OpInterpUtil::InferOpAttribute(op_expr, inputs));
  auto parallel_conf =
      std::make_shared<cfg::ParallelConf>(scope->device_parallel_desc_symbol()->parallel_conf());

  auto build_instruction = [&](const std::shared_ptr<InstructionsBuilder>& builder) {
    const auto& bn2blob_object =
        CHECK_JUST(OpInterpUtil::MakeBn2BlobObjectMap(op_expr.indexed_ibns(), inputs));
    const auto& boxing_util = *Global<std::shared_ptr<ForeignBoxingUtil>>::Get();
    CHECK_JUST(builder->StatelessCall(
        op_attribute, parallel_conf, bn2blob_object,
        std::bind(&ForeignBoxingUtil::BoxingTo, boxing_util.get(), _1, _2, _3)));
    for (int i = 0; i < outputs->size(); ++i) {
      const std::string& obn = op_expr.indexed_obns().at(i);
      auto t = CHECK_JUST(OpInterpUtil::BuildTensorFromBlobObject(bn2blob_object->at(obn)));
      outputs->at(i).swap(t);
    }
  };
  return LogicalRun(build_instruction);
}

Maybe<void> EagerInterpreter::ApplyImpl(const UserOpExpr& op_expr, const TensorTuple& inputs,
                                        TensorTuple* outputs) const {
  return NaiveInterpret(op_expr, inputs, outputs);
}

Maybe<void> EagerInterpreter::ApplyImpl(const VariableOpExpr& op_expr, const TensorTuple& inputs,
                                        TensorTuple* outputs) const {
  CHECK_EQ_OR_RETURN(inputs.size(), 0);
  CHECK_EQ_OR_RETURN(outputs->size(), 1);
  return NaiveInterpret(op_expr, inputs, outputs);
}

static Maybe<void> BuildAndRunMirroredCastInstruction(const BuiltinOpExpr& op_expr,
                                                      const TensorTuple& inputs,
                                                      TensorTuple* outputs) {
  const auto& op_attribute = JUST(OpInterpUtil::InferOpAttribute(op_expr, inputs));
  OpAttribute proto_op_attribute;
  op_attribute->ToProto(&proto_op_attribute);

  auto build_instruction = [&](const std::shared_ptr<InstructionsBuilder>& builder) {
    const auto& bn2blob_object =
        CHECK_JUST(OpInterpUtil::MakeBn2BlobObjectMap(op_expr.indexed_ibns(), inputs));
    const auto& in_blob_object = (*bn2blob_object)["in"];
    const auto& parallel_desc_symbol = in_blob_object->parallel_desc_symbol();
    const auto& op_arg_parallel_attr = CHECK_JUST(
        compatible_py::GetOpArgParallelAttribute(parallel_desc_symbol, proto_op_attribute, "out"));
    const auto& out_blob_object = CHECK_JUST(builder->MakeReferenceBlobObject(
        in_blob_object,
        std::make_shared<compatible_py::OpArgParallelAttribute>(*op_arg_parallel_attr)));
    auto t = CHECK_JUST(OpInterpUtil::BuildTensorFromBlobObject(out_blob_object));
    outputs->at(0).swap(t);
  };
  return LogicalRun(build_instruction);
}

Maybe<void> EagerInterpreter::ApplyImpl(const CastToMirroredOpExpr& op_expr,
                                        const TensorTuple& inputs, TensorTuple* outputs) const {
  return BuildAndRunMirroredCastInstruction(op_expr, inputs, outputs);
}

Maybe<void> EagerInterpreter::ApplyImpl(const CastFromMirroredOpExpr& op_expr,
                                        const TensorTuple& inputs, TensorTuple* outputs) const {
  return BuildAndRunMirroredCastInstruction(op_expr, inputs, outputs);
}

static Maybe<compatible_py::BlobObject> GetInBlobObject(
    const std::shared_ptr<InstructionsBuilder>& builder, const OpAttribute& op_attribute,
    const std::string& ibn,
    const HashMap<std::string, std::shared_ptr<compatible_py::BlobObject>>& bn2blob_object) {
  const auto& parallel_sig = op_attribute.parallel_signature().bn_in_op2parallel_desc_symbol_id();
  int symbol_id = parallel_sig.at(ibn);
  const auto& in_op_parallel_desc_sym = JUST(GetSymbol<cfg::ParallelConf, ParallelDesc>(symbol_id));
  const auto& in_op_arg_parallel_attr =
      JUST(compatible_py::GetOpArgParallelAttribute(in_op_parallel_desc_sym, op_attribute, ibn));
  const auto& origin_blob_object = bn2blob_object.at(ibn);
  return (*Global<std::shared_ptr<ForeignBoxingUtil>>::Get())
      ->BoxingTo(builder, origin_blob_object, in_op_arg_parallel_attr);
};

static Maybe<void> BuildAndRunDistributeSplitOrCloneInstruction(const BuiltinOpExpr& op_expr,
                                                                const TensorTuple& inputs,
                                                                TensorTuple* outputs) {
  const auto& op_attribute = JUST(OpInterpUtil::InferOpAttribute(op_expr, inputs));
  OpAttribute proto_op_attribute;
  op_attribute->ToProto(&proto_op_attribute);

  auto build_instruction = [&](const std::shared_ptr<InstructionsBuilder>& builder) {
    const auto& bn2blob_object =
        CHECK_JUST(OpInterpUtil::MakeBn2BlobObjectMap(op_expr.indexed_ibns(), inputs));
    const auto& logical_in_blob_object =
        CHECK_JUST(GetInBlobObject(builder, proto_op_attribute, "in", *bn2blob_object));
    const auto& physical_out_blob_objects =
        CHECK_JUST(builder->UnpackLogicalBlobToPhysicalBlobs(logical_in_blob_object));
    for (int i = 0; i < physical_out_blob_objects->size(); ++i) {
      auto t =
          CHECK_JUST(OpInterpUtil::BuildTensorFromBlobObject(physical_out_blob_objects->at(i)));
      outputs->at(i).swap(t);
    }
  };
  return LogicalRun(build_instruction);
}

Maybe<void> EagerInterpreter::ApplyImpl(const DistributeSplitOpExpr& op_expr,
                                        const TensorTuple& inputs, TensorTuple* outputs) const {
  return BuildAndRunDistributeSplitOrCloneInstruction(op_expr, inputs, outputs);
}

Maybe<void> EagerInterpreter::ApplyImpl(const DistributeCloneOpExpr& op_expr,
                                        const TensorTuple& inputs, TensorTuple* outputs) const {
  return BuildAndRunDistributeSplitOrCloneInstruction(op_expr, inputs, outputs);
}

static Maybe<void> BuildAndRunDistributeConcatAndAddInstruction(const BuiltinOpExpr& op_expr,
                                                                const TensorTuple& inputs,
                                                                TensorTuple* outputs) {
  const auto& op_attribute = JUST(OpInterpUtil::InferOpAttribute(op_expr, inputs));
  OpAttribute proto_op_attribute;
  op_attribute->ToProto(&proto_op_attribute);
  const auto& op_parallel_desc_sym = JUST(GetSymbol<cfg::ParallelConf, ParallelDesc>(
      proto_op_attribute.parallel_signature().op_parallel_desc_symbol_id()));
  const auto& op_arg_parallel_attr = JUST(
      compatible_py::GetOpArgParallelAttribute(op_parallel_desc_sym, proto_op_attribute, "out"));
  const auto& op_arg_blob_attr =
      JUST(compatible_py::GetOpArgBlobAttribute(proto_op_attribute, "out"));

  auto build_instruction = [&](const std::shared_ptr<InstructionsBuilder>& builder) {
    const auto& bn2blob_object =
        CHECK_JUST(OpInterpUtil::MakeBn2BlobObjectMap(op_expr.indexed_ibns(), inputs));
    int input_size = op_expr.indexed_ibns().size();
    std::vector<std::shared_ptr<compatible_py::BlobObject>> in_blob_objects(input_size);
    for (int i = 0; i < input_size; ++i) {
      in_blob_objects[i] = CHECK_JUST(
          GetInBlobObject(builder, proto_op_attribute, "in_" + std::to_string(i), *bn2blob_object));
    }
    const auto& physical_out_blob_object = CHECK_JUST(builder->PackPhysicalBlobsToLogicalBlob(
        in_blob_objects, op_arg_parallel_attr, op_arg_blob_attr));
    auto t = CHECK_JUST(OpInterpUtil::BuildTensorFromBlobObject(physical_out_blob_object));
    outputs->at(0).swap(t);
  };
  return LogicalRun(build_instruction);
}

Maybe<void> EagerInterpreter::ApplyImpl(const DistributeConcatOpExpr& op_expr,
                                        const TensorTuple& inputs, TensorTuple* outputs) const {
  return BuildAndRunDistributeConcatAndAddInstruction(op_expr, inputs, outputs);
}

Maybe<void> EagerInterpreter::ApplyImpl(const DistributeAddOpExpr& op_expr,
                                        const TensorTuple& inputs, TensorTuple* outputs) const {
  return BuildAndRunDistributeConcatAndAddInstruction(op_expr, inputs, outputs);
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
