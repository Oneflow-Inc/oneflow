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
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/framework/op_arg_util.h"
#include "oneflow/core/framework/scope_util.h"
#include "oneflow/core/framework/session_util.h"
#include "oneflow/core/framework/symbol_storage_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_impl.h"
#include "oneflow/core/framework/tensor_name_scope.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/framework/py_distribute.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/api/python/job_build/job_build_and_infer.h"
#include "oneflow/core/eager/foreign_boxing_util.h"

namespace oneflow {
namespace one {

namespace {

Maybe<Tensor> BuildTensorFromAttr(
    const std::shared_ptr<compatible_py::OpArgBlobAttribute>& blob_attr,
    const std::shared_ptr<compatible_py::OpArgParallelAttribute>& parallel_attr,
    const bool is_lazy) {
  const auto& dtype = JUST(DType::GetDTypeByDataType(DataType(blob_attr->get_dtype())));
  if (parallel_attr->is_mirrored()) {
    // TOOD(hjchen2): Use the right device.
    return static_cast<std::shared_ptr<Tensor>>(
        MirroredTensor::MakeTensor(blob_attr->shape(), dtype, std::make_shared<Device>("cpu", 0),
                                   is_lazy, /*requires_grad=*/false, /*is_leaf=*/false,
                                   /*retain_grad=*/false));
  } else {
    const auto& distribute =
        compatible_py::MakeDistribute(*(parallel_attr->sbp_parallel())).GetPtrOrThrow();
    return static_cast<std::shared_ptr<Tensor>>(ConsistentTensor::MakeTensor(
        blob_attr->shape(), dtype, distribute, parallel_attr->parallel_desc_symbol(), is_lazy,
        /*requires_grad=*/false, /*is_leaf=*/false, /*retain_grad=*/false));
  }
}

Maybe<Tensor> BuildTensorFromBlobObject(
    const std::shared_ptr<compatible_py::BlobObject>& blob_object) {
  const auto& blob_attr = blob_object->op_arg_blob_attr();
  const auto& parallel_attr = blob_object->op_arg_parallel_attr();
  const auto& tensor = JUST(BuildTensorFromAttr(blob_attr, parallel_attr, /*is_lazy=*/false));
  if (parallel_attr->is_mirrored()) {
    dynamic_cast<MirroredTensor*>(tensor.get())->set_blob_object(blob_object);
  } else {
    dynamic_cast<ConsistentTensor*>(tensor.get())->set_blob_object(blob_object);
  }
  return tensor;
}

}  // namespace

void OpExprInterpreter::ResetState() { state_.reset(new OpExprInterpState); }

Maybe<void> LazyInterpreter::Apply(const OpExpr* op_expr, const TensorTuple& inputs,
                                   TensorTuple& outputs) {
  ResetState();

#define APPLY_IF(op_type)                                             \
  if (const auto* op = dynamic_cast<const op_type##Expr*>(op_expr)) { \
    return Apply_(op, inputs, outputs);                               \
  }

  APPLY_IF(FunctionOp);
  APPLY_IF(BuiltinOp);
#undef APPLY_IF

  CHECK_OR_RETURN(false) << "The type " << op_expr->type()
                         << " has not been supported in LazyInterpreter::Apply.";
}

Maybe<void> LazyInterpreter::Apply_(const BuiltinOpExpr* op_expr, const TensorTuple& inputs,
                                    TensorTuple& outputs) {
  CHECK_EQ_OR_RETURN(inputs.size(), op_expr->input_num());
  const auto& scope = JUST(GetCurrentScope());
  auto op_conf = JUST(OpInterpUtil::GenBuiltinOpConf(op_expr));
  int64_t symbol_id = JUST(scope->symbol_id());
  op_conf->set_scope_symbol_id(symbol_id);
  if (!op_conf->has_device_tag()) {
    op_conf->set_device_tag(scope->device_parallel_desc_symbol()->device_tag());
  }
  for (int i = 0; i < inputs.size(); ++i) {
    const std::string& ibn = op_expr->indexed_ibns().at(i);
    const std::string& tensor_name = TensorNameScope::Global()->Lookup(inputs[i]);
    ReplaceInputLbnInOpCustomizedConf(op_conf.get(), ibn, tensor_name);
  }
  const auto& op_attribute = JUST(OpInterpUtil::AddBuiltinOpAndInferOpAttribute(
      *op_conf, context()->is_mirrored_strategy_enabled));
  OpAttribute proto_op_attribute;
  op_attribute->ToProto(&proto_op_attribute);

  int64_t parallel_desc_sym_id = JUST(scope->GetParallelDescSymbolId(*op_conf));
  const std::shared_ptr<ParallelDesc>& blob_parallel_desc_sym =
      JUST(GetSymbol<cfg::ParallelConf, ParallelDesc>(parallel_desc_sym_id));

  // Check outputs num and setup output tensor properties.
  CHECK_EQ_OR_RETURN(outputs.size(), op_expr->output_num());
  for (int i = 0; i < op_expr->output_num(); ++i) {
    const std::string& obn = op_expr->indexed_obns().at(i);
    const auto& parallel_attr = JUST(
        compatible_py::GetOpArgParallelAttribute(blob_parallel_desc_sym, proto_op_attribute, obn));
    const auto& blob_attr = JUST(compatible_py::GetOpArgBlobAttribute(proto_op_attribute, obn));
    if (!(outputs[i].get())) {
      outputs[i] = JUST(BuildTensorFromAttr(blob_attr, parallel_attr, /*is_lazy=*/true));
    } else {
      // TODO(hjchen2) Set shape, dtype, ...
    }
    TensorNameScope::Global()->Record(outputs[i], op_expr->op_name() + "/" + obn);
  }
  return Maybe<void>::Ok();
}

Maybe<void> LazyInterpreter::Apply_(const FunctionOpExpr* op_expr, const TensorTuple& inputs,
                                    TensorTuple& outputs) {
  // TODO
  UNIMPLEMENTED();
  return Maybe<void>::Ok();
}

Maybe<void> EagerInterpreter::Apply(const OpExpr* op_expr, const TensorTuple& inputs,
                                    TensorTuple& outputs) {
  ResetState();

#define APPLY_IF(op_type)                                             \
  if (const auto* op = dynamic_cast<const op_type##Expr*>(op_expr)) { \
    return Apply_(op, inputs, outputs);                               \
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

  CHECK_OR_RETURN(false) << "The type " << op_expr->type()
                         << " has not been supported in EagerInterpreter::Apply.";
}

static Maybe<void> NaiveInterpret(const BuiltinOpExpr* op_expr, const TensorTuple& inputs,
                                  TensorTuple& outputs,
                                  const std::shared_ptr<cfg::OpAttribute>& op_attribute,
                                  const std::shared_ptr<cfg::ParallelConf>& parallel_conf) {
  auto BuildInstruction = [&](const std::shared_ptr<InstructionsBuilder>& builder) {
    const auto& bn2blob_object =
        CHECK_JUST(OpInterpUtil::MakeBn2BlobObjectMap(op_expr->indexed_ibns(), inputs));
    CHECK_JUST(builder->NoBoxingStatelessCall(op_attribute, parallel_conf, bn2blob_object));
    for (int i = 0; i < outputs.size(); ++i) {
      const std::string& obn = op_expr->indexed_obns().at(i);
      outputs[i] = CHECK_JUST(BuildTensorFromBlobObject(bn2blob_object->at(obn)));
    }
  };
  return LogicalRun(BuildInstruction);
}

Maybe<void> EagerInterpreter::Apply_(const UserOpExpr* op_expr, const TensorTuple& inputs,
                                     TensorTuple& outputs) {
  const auto& scope = JUST(GetCurrentScope());
  const auto& op_attribute = JUST(OpInterpUtil::InferOpAttribute(op_expr, scope, inputs));
  auto parallel_conf =
      std::make_shared<cfg::ParallelConf>(scope->device_parallel_desc_symbol()->parallel_conf());
  return NaiveInterpret(op_expr, inputs, outputs, op_attribute, parallel_conf);
}

Maybe<void> EagerInterpreter::Apply_(const VariableOpExpr* op_expr, const TensorTuple& inputs,
                                     TensorTuple& outputs) {
  CHECK_EQ_OR_RETURN(inputs.size(), 0);
  CHECK_EQ_OR_RETURN(outputs.size(), 1);
  const auto& job_name = JUST(JobBuildAndInferCtx_GetCurrentJobName());
  const auto& session = JUST(GetDefaultSession());
  const std::string variable_name = session->GetJobNameScopePrefix(*job_name) + op_expr->op_name();

  std::shared_ptr<Tensor> global_blob, job_blob;
  std::tie(global_blob, job_blob) =
      session->TryGetVariableBlobOfJobFromStash(*job_name, variable_name);
  if (global_blob.get()) {
    outputs[0] = global_blob;
    return Maybe<void>::Ok();
  }
  const auto& scope = JUST(GetCurrentScope());
  const auto& op_attribute = JUST(OpInterpUtil::InferOpAttribute(op_expr, scope, inputs));
  auto parallel_conf =
      std::make_shared<cfg::ParallelConf>(scope->device_parallel_desc_symbol()->parallel_conf());
  JUST(NaiveInterpret(op_expr, inputs, outputs, op_attribute, parallel_conf));
  OpAttribute proto_op_attribute;
  op_attribute->ToProto(&proto_op_attribute);

  const auto& blob_attr = JUST(compatible_py::GetOpArgBlobAttribute(proto_op_attribute, "out"));
  auto parallel_desc =
      std::make_shared<ParallelDesc>(scope->device_parallel_desc_symbol()->parallel_conf());
  const auto& parallel_attr =
      JUST(compatible_py::GetOpArgParallelAttribute(parallel_desc, proto_op_attribute, "out"));
  outputs[0] = JUST(BuildTensorFromAttr(blob_attr, parallel_attr, /*is_lazy=*/false));
  return OpInterpUtil::InitVariableOutputBlob(session, outputs[0], proto_op_attribute);
}

static Maybe<std::function<void(const std::shared_ptr<InstructionsBuilder>&)>>
BuildMirroredCastInstruction(const BuiltinOpExpr* op_expr, const TensorTuple& inputs,
                             TensorTuple& outputs, const OpExprInterpContext* ctx) {
  const auto& scope = JUST(GetCurrentScope());
  const auto& op_attribute = JUST(OpInterpUtil::InferOpAttribute(op_expr, scope, inputs));
  auto build_instruction = [&, scope,
                            op_attribute](const std::shared_ptr<InstructionsBuilder>& builder) {
    const auto& bn2blob_object =
        CHECK_JUST(OpInterpUtil::MakeBn2BlobObjectMap(op_expr->indexed_ibns(), inputs));
    const auto& in_blob_object = (*bn2blob_object)["in"];
    const auto& parallel_desc_symbol = in_blob_object->parallel_desc_symbol();
    OpAttribute proto_op_attribute;
    op_attribute->ToProto(&proto_op_attribute);
    const auto& op_arg_parallel_attr = CHECK_JUST(
        compatible_py::GetOpArgParallelAttribute(parallel_desc_symbol, proto_op_attribute, "out"));
    const auto& out_blob_object = CHECK_JUST(builder->MakeReferenceBlobObject(
        in_blob_object,
        std::make_shared<compatible_py::OpArgParallelAttribute>(*op_arg_parallel_attr)));
    outputs[0] = CHECK_JUST(BuildTensorFromBlobObject(out_blob_object));
  };
  return std::function<void(const std::shared_ptr<InstructionsBuilder>&)>(build_instruction);
}

Maybe<void> EagerInterpreter::Apply_(const CastToMirroredOpExpr* op_expr, const TensorTuple& inputs,
                                     TensorTuple& outputs) {
  auto build_instruction = JUST(BuildMirroredCastInstruction(op_expr, inputs, outputs, context()));
  return LogicalRun(*build_instruction);
}

Maybe<void> EagerInterpreter::Apply_(const CastFromMirroredOpExpr* op_expr,
                                     const TensorTuple& inputs, TensorTuple& outputs) {
  auto build_instruction = JUST(BuildMirroredCastInstruction(op_expr, inputs, outputs, context()));
  return LogicalRun(*build_instruction);
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

static Maybe<std::function<void(const std::shared_ptr<InstructionsBuilder>&)>>
BuildDistributeSplitOrCloneInstruction(const BuiltinOpExpr* op_expr, const TensorTuple& inputs,
                                       TensorTuple& outputs, const OpExprInterpContext* ctx) {
  const auto& scope = JUST(GetCurrentScope());
  const auto& op_attribute = JUST(OpInterpUtil::InferOpAttribute(op_expr, scope, inputs));
  auto build_instruction = [&, scope,
                            op_attribute](const std::shared_ptr<InstructionsBuilder>& builder) {
    OpAttribute proto_op_attribute;
    op_attribute->ToProto(&proto_op_attribute);
    const auto& bn2blob_object =
        CHECK_JUST(OpInterpUtil::MakeBn2BlobObjectMap(op_expr->indexed_ibns(), inputs));
    const auto& logical_in_blob_object =
        CHECK_JUST(GetInBlobObject(builder, proto_op_attribute, "in", *bn2blob_object));
    const auto& physical_out_blob_objects =
        CHECK_JUST(builder->UnpackLogicalBlobToPhysicalBlobs(logical_in_blob_object));
    for (int i = 0; i < physical_out_blob_objects->size(); ++i) {
      outputs[i] = CHECK_JUST(BuildTensorFromBlobObject(physical_out_blob_objects->at(i)));
    }
  };
  return std::function<void(const std::shared_ptr<InstructionsBuilder>&)>(build_instruction);
}

Maybe<void> EagerInterpreter::Apply_(const DistributeSplitOpExpr* op_expr,
                                     const TensorTuple& inputs, TensorTuple& outputs) {
  auto build_instruction =
      JUST(BuildDistributeSplitOrCloneInstruction(op_expr, inputs, outputs, context()));
  return LogicalRun(*build_instruction);
}

Maybe<void> EagerInterpreter::Apply_(const DistributeCloneOpExpr* op_expr,
                                     const TensorTuple& inputs, TensorTuple& outputs) {
  auto build_instruction =
      JUST(BuildDistributeSplitOrCloneInstruction(op_expr, inputs, outputs, context()));
  return LogicalRun(*build_instruction);
}

static Maybe<std::function<void(const std::shared_ptr<InstructionsBuilder>&)>>
BuildDistributeConcatAndAddInstruction(const BuiltinOpExpr* op_expr, const TensorTuple& inputs,
                                       TensorTuple& outputs, const OpExprInterpContext* ctx) {
  const auto& scope = JUST(GetCurrentScope());
  const auto& op_attribute = JUST(OpInterpUtil::InferOpAttribute(op_expr, scope, inputs));
  auto build_instruction = [&, scope,
                            op_attribute](const std::shared_ptr<InstructionsBuilder>& builder) {
    OpAttribute proto_op_attribute;
    op_attribute->ToProto(&proto_op_attribute);
    const auto& op_parallel_desc_sym = CHECK_JUST(GetSymbol<cfg::ParallelConf, ParallelDesc>(
        proto_op_attribute.parallel_signature().op_parallel_desc_symbol_id()));
    const auto& op_arg_parallel_attr = CHECK_JUST(
        compatible_py::GetOpArgParallelAttribute(op_parallel_desc_sym, proto_op_attribute, "out"));
    const auto& op_arg_blob_attr =
        CHECK_JUST(compatible_py::GetOpArgBlobAttribute(proto_op_attribute, "out"));
    int input_size = op_expr->indexed_ibns().size();
    const auto& bn2blob_object =
        CHECK_JUST(OpInterpUtil::MakeBn2BlobObjectMap(op_expr->indexed_ibns(), inputs));
    std::vector<std::shared_ptr<compatible_py::BlobObject>> in_blob_objects(input_size);
    for (int i = 0; i < input_size; ++i) {
      in_blob_objects[i] = CHECK_JUST(
          GetInBlobObject(builder, proto_op_attribute, "in_" + std::to_string(i), *bn2blob_object));
    }
    const auto& physical_out_blob_object = CHECK_JUST(builder->PackPhysicalBlobsToLogicalBlob(
        in_blob_objects, op_arg_parallel_attr, op_arg_blob_attr));
    outputs[0] = CHECK_JUST(BuildTensorFromBlobObject(physical_out_blob_object));
  };
  return std::function<void(const std::shared_ptr<InstructionsBuilder>&)>(build_instruction);
}

Maybe<void> EagerInterpreter::Apply_(const DistributeConcatOpExpr* op_expr,
                                     const TensorTuple& inputs, TensorTuple& outputs) {
  auto build_instruction =
      JUST(BuildDistributeConcatAndAddInstruction(op_expr, inputs, outputs, context()));
  return LogicalRun(*build_instruction);
}

Maybe<void> EagerInterpreter::Apply_(const DistributeAddOpExpr* op_expr, const TensorTuple& inputs,
                                     TensorTuple& outputs) {
  auto build_instruction =
      JUST(BuildDistributeConcatAndAddInstruction(op_expr, inputs, outputs, context()));
  return LogicalRun(*build_instruction);
}

Maybe<void> EagerInterpreter::Apply_(const FunctionOpExpr* op_expr, const TensorTuple& inputs,
                                     TensorTuple& outputs) {
  // TODO(hjchen2)
  UNIMPLEMENTED();
  return Maybe<void>::Ok();
}

Maybe<void> AutogradInterpreter::Apply(const OpExpr* op_expr, const TensorTuple& inputs,
                                       TensorTuple& outputs) {
  // TODO(hjchen2)
  return normal_interp_->Apply(op_expr, inputs, outputs);
}

}  // namespace one
}  // namespace oneflow
