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
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/framework/op_arg_util.h"
#include "oneflow/core/framework/scope_util.h"
#include "oneflow/core/framework/session_util.h"
#include "oneflow/core/framework/symbol_storage_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_impl.h"
#include "oneflow/core/framework/tensor_name_scope.h"
#include "oneflow/core/framework/py_distribute.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/api/python/job_build/job_build_and_infer.h"
#include "oneflow/core/eager/foreign_boxing_util.h"

namespace oneflow {
namespace one {

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
  auto scope = GetCurrentScope().GetPtrOrThrow();
  auto op_conf = OpInterpUtil::GenBuiltinOpConf(op_expr);
  int64_t symbol_id = scope->symbol_id().GetOrThrow();
  op_conf->set_scope_symbol_id(symbol_id);
  if (!op_conf->has_device_tag()) {
    op_conf->set_device_tag(scope->device_parallel_desc_symbol()->device_tag());
  }
  for (int i = 0; i < inputs.size(); ++i) {
    const std::string& ibn = op_expr->indexed_ibns().at(i);
    const std::string& tensor_name = TensorNameScope::Global()->Lookup(inputs[i]);
    ReplaceInputLbnInOpCustomizedConf(op_conf.get(), ibn, tensor_name);
  }
  auto op_attribute = OpInterpUtil::AddBuiltinOpAndInferOpAttribute(
      *op_conf, context()->is_mirrored_strategy_enabled);
  OpAttribute proto_op_attribute;
  op_attribute->ToProto(&proto_op_attribute);

  // Check outputs num and setup output tensor properties.
  CHECK_EQ(outputs.size(), op_expr->output_num());
  for (int i = 0; i < op_expr->output_num(); ++i) {
    const auto& bn_in_op2blob_desc =
        proto_op_attribute.logical_blob_desc_signature().bn_in_op2blob_desc();
    const std::string& obn = op_expr->indexed_obns().at(i);
    BlobDesc blob_desc(bn_in_op2blob_desc.at(obn));
    // outputs[i]->set_blob_desc(blob_desc);
    TensorNameScope::Global()->Record(outputs[i], op_expr->op_name() + "/" + obn);
  }
}

void LazyInterpreter::Apply_(const FunctionOpExpr* op_expr, const TensorList& inputs,
                             TensorList& outputs, const OpExprInterpState* state) {
  // TODO
  UNIMPLEMENTED();
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

static std::shared_ptr<HashMap<std::string, std::shared_ptr<compatible_py::BlobObject>>>
MakeBn2BlobObjectMap(const BuiltinOpExpr* op_expr, const TensorList& inputs,
                     const TensorList& outputs) {
  auto* bn2blob_object(new HashMap<std::string, std::shared_ptr<compatible_py::BlobObject>>{});
  for (int i = 0; i < inputs.size(); ++i) {
    const auto& ibn = op_expr->indexed_ibns().at(i);
    bn2blob_object->emplace(ibn, dynamic_cast<DeterminedTensor*>(inputs[i].get())->blob_object());
  }
  return std::shared_ptr<HashMap<std::string, std::shared_ptr<compatible_py::BlobObject>>>(
      bn2blob_object);
}

std::shared_ptr<Tensor> BuildTensorFromBlobObject(
    const std::shared_ptr<compatible_py::BlobObject>& blob_object) {
  std::shared_ptr<Tensor> tensor;
  const auto& blob_attr = blob_object->op_arg_blob_attr();
  const auto& parallel_attr = blob_object->op_arg_parallel_attr();
  DataType dtype = DataType(blob_attr->get_dtype());
  if (parallel_attr->is_mirrored()) {
    // TOOD(hjchen2): Use the right device.
    auto impl = std::make_shared<EagerMirroredTensorImpl>(blob_attr->shape(), dtype,
                                                          std::make_shared<Device>("cpu", 0));
    tensor = std::make_shared<MirroredTensor>(impl);
  } else {
    auto distribute =
        compatible_py::MakeDistribute(*(parallel_attr->sbp_parallel())).GetPtrOrThrow();
    auto impl = std::make_shared<EagerConsistentTensorImpl>(blob_attr->shape(), dtype, distribute,
                                                            parallel_attr->parallel_desc_symbol());
    tensor = std::make_shared<ConsistentTensor>(impl);
  }
  dynamic_cast<DeterminedTensor*>(tensor.get())->set_blob_object(blob_object);
  return tensor;
}

static void NaiveInterpret(const BuiltinOpExpr* op_expr, const TensorList& inputs,
                           TensorList& outputs,
                           const std::shared_ptr<cfg::OpAttribute>& op_attribute,
                           const std::shared_ptr<cfg::ParallelConf>& parallel_conf) {
  auto BuildInstruction = [&](const std::shared_ptr<InstructionsBuilder>& builder) {
    auto bn2blob_object = MakeBn2BlobObjectMap(op_expr, inputs, outputs);
    builder->NoBoxingStatelessCall(op_attribute, parallel_conf, bn2blob_object);
    for (int i = 0; i < outputs.size(); ++i) {
      const std::string& obn = op_expr->indexed_obns().at(i);
      outputs[i] = BuildTensorFromBlobObject(bn2blob_object->at(obn));
    }
  };
  void(LogicalRun(BuildInstruction).GetOrThrow());
}

void EagerInterpreter::Apply_(const UserOpExpr* op_expr, const TensorList& inputs,
                              TensorList& outputs, const OpExprInterpState* state) {
  auto scope = GetCurrentScope().GetPtrOrThrow();
  auto op_attribute = OpInterpUtil::AddBuiltinOpAndInferOpAttribute(
      op_expr, scope, context()->is_mirrored_strategy_enabled);
  auto parallel_conf =
      std::make_shared<cfg::ParallelConf>(scope->device_parallel_desc_symbol()->parallel_conf());
  NaiveInterpret(op_expr, inputs, outputs, op_attribute, parallel_conf);
}

void EagerInterpreter::Apply_(const VariableOpExpr* op_expr, const TensorList& inputs,
                              TensorList& outputs, const OpExprInterpState* state) {
  CHECK_EQ(inputs.size(), 0);
  CHECK_EQ(outputs.size(), 1);
  const std::string job_name = JobBuildAndInferCtx_GetCurrentJobName().GetOrThrow();
  auto session = GetDefaultSession().GetPtrOrThrow();
  const std::string variable_name = session->GetJobNameScopePrefix(job_name) + op_expr->op_name();

  std::shared_ptr<Tensor> global_blob, job_blob;
  std::tie(global_blob, job_blob) =
      session->TryGetVariableBlobOfJobFromStash(job_name, variable_name);
  if (global_blob.get()) {
    outputs[0] = global_blob;
    return;
  }
  auto scope = GetCurrentScope().GetPtrOrThrow();
  auto op_attribute = OpInterpUtil::AddBuiltinOpAndInferOpAttribute(
      op_expr, scope, context()->is_mirrored_strategy_enabled);
  auto parallel_conf =
      std::make_shared<cfg::ParallelConf>(scope->device_parallel_desc_symbol()->parallel_conf());
  NaiveInterpret(op_expr, inputs, outputs, op_attribute, parallel_conf);
  OpAttribute proto_op_attribute;
  op_attribute->ToProto(&proto_op_attribute);

  auto blob_attr = compatible_py::GetOpArgBlobAttribute(proto_op_attribute, "out").GetPtrOrThrow();
  DataType dtype = DataType(blob_attr->get_dtype());
  const auto& mirrored_sig_map =
      proto_op_attribute.mirrored_signature().bn_in_op2opt_mirrored_parallel();
  if (mirrored_sig_map.at("out").has_mirrored_parallel()) {
    auto impl = std::make_shared<EagerMirroredTensorImpl>(blob_attr->shape(), dtype,
                                                          std::make_shared<Device>("cpu", 0));
    outputs[0].reset(new MirroredTensor(impl));
  } else {
    auto parallel_desc =
        std::make_shared<ParallelDesc>(scope->device_parallel_desc_symbol()->parallel_conf());
    auto impl = std::make_shared<EagerConsistentTensorImpl>(
        blob_attr->shape(), dtype, compatible_py::GlobalAutoDistribute(), parallel_desc);
    outputs[0].reset(new ConsistentTensor(impl));
  }
  OpInterpUtil::InitVariableOutputBlob(session, outputs[0], proto_op_attribute);
}

static std::function<void(const std::shared_ptr<InstructionsBuilder>& builder)>
BuildMirroredCastInstruction(const BuiltinOpExpr* op_expr, const TensorList& inputs,
                             TensorList& outputs, const OpExprInterpContext* ctx) {
  auto scope = GetCurrentScope().GetPtrOrThrow();
  auto op_attribute = OpInterpUtil::AddBuiltinOpAndInferOpAttribute(
      op_expr, scope, ctx->is_mirrored_strategy_enabled);
  auto BuildInstruction = [&](const std::shared_ptr<InstructionsBuilder>& builder) {
    auto bn2blob_object = MakeBn2BlobObjectMap(op_expr, inputs, outputs);
    const auto& in_blob_object = (*bn2blob_object)["in"];
    const auto& parallel_desc_symbol = in_blob_object->parallel_desc_symbol();
    OpAttribute proto_op_attribute;
    op_attribute->ToProto(&proto_op_attribute);
    const auto& op_arg_parallel_attr =
        compatible_py::GetOpArgParallelAttribute(parallel_desc_symbol, proto_op_attribute, "out")
            .GetOrThrow();
    auto out_blob_object =
        builder
            ->MakeReferenceBlobObject(
                in_blob_object,
                std::make_shared<compatible_py::OpArgParallelAttribute>(op_arg_parallel_attr))
            .GetPtrOrThrow();
    outputs[0] = BuildTensorFromBlobObject(out_blob_object);
  };
  return BuildInstruction;
}

void EagerInterpreter::Apply_(const CastToMirroredOpExpr* op_expr, const TensorList& inputs,
                              TensorList& outputs, const OpExprInterpState* state) {
  auto BuildInstruction = BuildMirroredCastInstruction(op_expr, inputs, outputs, context());
  LogicalRun(BuildInstruction).GetOrThrow();
}

void EagerInterpreter::Apply_(const CastFromMirroredOpExpr* op_expr, const TensorList& inputs,
                              TensorList& outputs, const OpExprInterpState* state) {
  auto BuildInstruction = BuildMirroredCastInstruction(op_expr, inputs, outputs, context());
  LogicalRun(BuildInstruction).GetOrThrow();
}

static std::shared_ptr<compatible_py::BlobObject> GetInBlobObject(
    const std::shared_ptr<InstructionsBuilder>& builder, const OpAttribute& op_attribute,
    const std::string& ibn,
    const HashMap<std::string, std::shared_ptr<compatible_py::BlobObject>>& bn2blob_object) {
  const auto& parallel_sig = op_attribute.parallel_signature().bn_in_op2parallel_desc_symbol_id();
  int symbol_id = parallel_sig.at(ibn);
  auto in_op_parallel_desc_sym =
      GetSymbol<cfg::ParallelConf, ParallelDesc>(symbol_id).GetPtrOrThrow();
  const auto& in_op_arg_parallel_attr =
      compatible_py::GetOpArgParallelAttribute(in_op_parallel_desc_sym, op_attribute, ibn)
          .GetPtrOrThrow();
  auto origin_blob_object = bn2blob_object.at(ibn);
  return Global<ForeignBoxingUtil>::Get()->BoxingTo(builder, origin_blob_object,
                                                    in_op_arg_parallel_attr);
};

static std::function<void(const std::shared_ptr<InstructionsBuilder>& builder)>
BuildDistributeSplitOrCloneInstruction(const BuiltinOpExpr* op_expr, const TensorList& inputs,
                                       TensorList& outputs, const OpExprInterpContext* ctx) {
  auto scope = GetCurrentScope().GetPtrOrThrow();
  auto op_attribute = OpInterpUtil::AddBuiltinOpAndInferOpAttribute(
      op_expr, scope, ctx->is_mirrored_strategy_enabled);
  OpAttribute proto_op_attribute;
  op_attribute->ToProto(&proto_op_attribute);
  auto BuildInstruction = [&](const std::shared_ptr<InstructionsBuilder>& builder) {
    auto bn2blob_object = MakeBn2BlobObjectMap(op_expr, inputs, outputs);
    auto logical_in_blob_object =
        GetInBlobObject(builder, proto_op_attribute, "in", *bn2blob_object);
    auto physical_out_blob_objects =
        builder->UnpackLogicalBlobToPhysicalBlobs(logical_in_blob_object).GetOrThrow();
    for (int i = 0; i < physical_out_blob_objects.size(); ++i) {
      outputs[i] = BuildTensorFromBlobObject(physical_out_blob_objects[i]);
    }
  };
  return BuildInstruction;
}

void EagerInterpreter::Apply_(const DistributeSplitOpExpr* op_expr, const TensorList& inputs,
                              TensorList& outputs, const OpExprInterpState* state) {
  auto BuildInstruction =
      BuildDistributeSplitOrCloneInstruction(op_expr, inputs, outputs, context());
  LogicalRun(BuildInstruction).GetOrThrow();
}

void EagerInterpreter::Apply_(const DistributeCloneOpExpr* op_expr, const TensorList& inputs,
                              TensorList& outputs, const OpExprInterpState* state) {
  auto BuildInstruction =
      BuildDistributeSplitOrCloneInstruction(op_expr, inputs, outputs, context());
  LogicalRun(BuildInstruction).GetOrThrow();
}

static std::function<void(const std::shared_ptr<InstructionsBuilder>& builder)>
BuildDistributeConcatAndAddInstruction(const BuiltinOpExpr* op_expr, const TensorList& inputs,
                                       TensorList& outputs, const OpExprInterpContext* ctx) {
  auto scope = GetCurrentScope().GetPtrOrThrow();
  auto op_attribute = OpInterpUtil::AddBuiltinOpAndInferOpAttribute(
      op_expr, scope, ctx->is_mirrored_strategy_enabled);
  OpAttribute proto_op_attribute;
  op_attribute->ToProto(&proto_op_attribute);
  auto op_parallel_desc_sym =
      GetSymbol<cfg::ParallelConf, ParallelDesc>(
          proto_op_attribute.parallel_signature().op_parallel_desc_symbol_id())
          .GetPtrOrThrow();
  auto op_arg_parallel_attr =
      compatible_py::GetOpArgParallelAttribute(op_parallel_desc_sym, proto_op_attribute, "out")
          .GetPtrOrThrow();
  auto op_arg_blob_attr =
      compatible_py::GetOpArgBlobAttribute(proto_op_attribute, "out").GetPtrOrThrow();
  auto BuildInstruction = [&](const std::shared_ptr<InstructionsBuilder>& builder) {
    int input_size = op_expr->indexed_ibns().size();
    auto bn2blob_object = MakeBn2BlobObjectMap(op_expr, inputs, outputs);
    std::vector<std::shared_ptr<compatible_py::BlobObject>> in_blob_objects(input_size);
    for (int i = 0; i < input_size; ++i) {
      in_blob_objects[i] =
          GetInBlobObject(builder, proto_op_attribute, "in_" + std::to_string(i), *bn2blob_object);
    }
    auto physical_out_blob_object = builder
                                        ->PackPhysicalBlobsToLogicalBlob(
                                            in_blob_objects, op_arg_parallel_attr, op_arg_blob_attr)
                                        .GetPtrOrThrow();
    outputs[0] = BuildTensorFromBlobObject(physical_out_blob_object);
  };
  return BuildInstruction;
}

void EagerInterpreter::Apply_(const DistributeConcatOpExpr* op_expr, const TensorList& inputs,
                              TensorList& outputs, const OpExprInterpState* state) {
  auto BuildInstruction =
      BuildDistributeConcatAndAddInstruction(op_expr, inputs, outputs, context());
  LogicalRun(BuildInstruction).GetOrThrow();
}

void EagerInterpreter::Apply_(const DistributeAddOpExpr* op_expr, const TensorList& inputs,
                              TensorList& outputs, const OpExprInterpState* state) {
  auto BuildInstruction =
      BuildDistributeConcatAndAddInstruction(op_expr, inputs, outputs, context());
  LogicalRun(BuildInstruction).GetOrThrow();
}

void EagerInterpreter::Apply_(const FunctionOpExpr* op_expr, const TensorList& inputs,
                              TensorList& outputs, const OpExprInterpState* state) {
  // TODO(hjchen2)
  UNIMPLEMENTED();
}

void AutogradInterpreter::Apply(const OpExpr* op_expr, const TensorList& inputs,
                                TensorList& outputs, const OpExprInterpState* state) {
  // TODO(hjchen2)
  UNIMPLEMENTED();
}

}  // namespace one
}  // namespace oneflow
