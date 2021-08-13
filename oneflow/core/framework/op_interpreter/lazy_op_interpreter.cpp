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
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/framework/multi_client_session_context.h"
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
#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"
#include "oneflow/core/vm/vm_util.h"

namespace oneflow {

namespace one {

std::string GetDeviceTagOfTensor(const std::shared_ptr<Tensor>& tensor) {
  if (tensor->is_cuda()) {
    return "gpu";
  } else {
    return "cpu";
  }
}

std::string GetDeviceTagByDeviceTypeStr(const std::string& device_type) {
  if (device_type == "cuda") {
    return "gpu";
  } else {
    return "cpu";
  }
}

bool GetIsDynamicOfTensor(const std::shared_ptr<Tensor>& tensor) {
  if (tensor->is_consistent()) {
    return false;
  } else {
    return true;
  }
}

Maybe<void> GenParallelDistributionByTensor(ParallelDistribution* nd_sbp,
                                            const std::shared_ptr<Tensor>& tensor) {
  nd_sbp->clear_sbp_parallel();
  if (tensor->is_local()) {
    // NOTE(chengcheng):
    //   OneFlow Lazy is always consistent. LocalTensor is a special case of ConsistentTensor which
    //   placement is only this rank, and SbpParallel is Broadcast.
    nd_sbp->add_sbp_parallel()->mutable_broadcast_parallel();
  } else {
    JUST(tensor->nd_sbp())->ToProto(nd_sbp);
  }
  return Maybe<void>::Ok();
}

Maybe<void> GenVariableOpConfParallelDistributionStringByTensor(
    VariableOpConf* var_conf, const std::shared_ptr<Tensor>& tensor) {
  var_conf->clear_nd_sbp();
  if (tensor->is_local()) {
    cfg::SbpParallel broadcast;
    broadcast.mutable_broadcast_parallel();
    var_conf->add_nd_sbp(SbpParallelToString(broadcast));
  } else {
    const cfg::ParallelDistribution& nd_sbp = *JUST(tensor->nd_sbp());
    for (const auto& sbp_parallel : nd_sbp.sbp_parallel()) {
      var_conf->add_nd_sbp(SbpParallelToString(sbp_parallel));
    }
  }
  return Maybe<void>::Ok();
}

Maybe<const ParallelDesc> GetParallelDescOfTensor(const std::shared_ptr<Tensor>& tensor) {
  if (tensor->is_local()) {
    return JUST(tensor->device())->parallel_desc_ptr();
  } else {
    return JUST(tensor->parallel_desc()).shared_from_symbol();
  }
}

Maybe<Scope> NewScopeWithParallelConfAndCurScope(
    const std::shared_ptr<cfg::ParallelConf>& parallel_conf) {
  std::shared_ptr<Scope> new_scope;
  const auto& old_scope = JUST(GetCurrentScope());
  JUST(PhysicalRun([&](InstructionsBuilder* builder) -> Maybe<void> {
    new_scope = JUST(builder->BuildScopeWithNewParallelConf(old_scope, parallel_conf));
    return Maybe<void>::Ok();
  }));
  // NOTE(chengcheng): need sync vm for get scope right now
  JUST(vm::MultiClientSync());
  CHECK_OR_RETURN(new_scope);
  return new_scope;
}

Maybe<Scope> NewScopeWithParallelDescByTensor(const std::shared_ptr<Tensor>& tensor) {
  std::shared_ptr<cfg::ParallelConf> parallel_conf = std::make_shared<cfg::ParallelConf>();
  parallel_conf->InitFromProto(JUST(GetParallelDescOfTensor(tensor))->parallel_conf());
  return NewScopeWithParallelConfAndCurScope(parallel_conf);
}

Maybe<void> LazyInterpreter::ApplyImpl(const FeedInputOpExpr& op_expr, const TensorTuple& inputs,
                                       TensorTuple* outputs, const OpExprInterpContext& ctx) const {
  // NOTE(chengcheng): inputs[0] is the EagerTensor
  CHECK_EQ_OR_RETURN(inputs.size(), 1);
  CHECK_EQ_OR_RETURN(op_expr.input_size(), 1);
  const std::shared_ptr<Tensor>& input_tensor = inputs.at(0);
  CHECK_OR_RETURN(input_tensor->is_eager());

  std::shared_ptr<Scope> scope = JUST(NewScopeWithParallelDescByTensor(input_tensor));

  OperatorConf op_conf;
  op_conf.set_name(op_expr.op_name());  // construct by python nn.Graph
  op_conf.set_scope_symbol_id(JUST(scope->symbol_id()));
  op_conf.set_device_tag(GetDeviceTagOfTensor(input_tensor));
  // NOTE(chengcheng):
  //   We contruct InputOpConf instead of FeedInputOpConf because FeedInputOpExpr JUST for getting
  //   input EagerTensor.
  InputOpConf* input_conf = op_conf.mutable_input_conf();
  input_conf->set_out("out");
  InterfaceBlobConf* blob_conf = input_conf->mutable_blob_conf();

  input_tensor->shape()->ToProto(blob_conf->mutable_shape());
  blob_conf->set_data_type(input_tensor->dtype());
  // NOTE(chengcheng): is_dynamic true has conflict in consistent lazy job even if world size 1.
  //     this flag will be removed in the future.
  // blob_conf->set_is_dynamic(GetIsDynamicOfTensor(input_tensor));
  blob_conf->set_is_dynamic(false);
  JUST(GenParallelDistributionByTensor(blob_conf->mutable_nd_sbp(), input_tensor));

  auto infer_ctx = JUST(GetCurInferCtx());
  OpAttribute op_attr = *JUST(infer_ctx->AddAndInferConsistentOp(op_conf));

  VLOG(2) << "Lazy nn.Graph name " << infer_ctx->job().job_conf().job_name() << " add op : \n"
          << op_conf.DebugString() << std::endl;
  VLOG(3) << "Lazy nn.Graph name " << infer_ctx->job().job_conf().job_name()
          << " infer and and op attr : \n"
          << op_attr.DebugString() << std::endl;

  int64_t parallel_desc_sym_id = JUST(scope->GetParallelDescSymbolId(op_conf));
  const std::shared_ptr<ParallelDesc>& blob_parallel_desc_sym =
      JUST(GetSymbol<cfg::ParallelConf, ParallelDesc>(parallel_desc_sym_id));

  // Check outputs num and setup output tensor properties.
  CHECK_EQ_OR_RETURN(outputs->size(), 1);
  CHECK_EQ_OR_RETURN(op_expr.output_size(), 1);

  const std::string obn = "out";  // NOTE(chengcheng): obn is NOT op_expr.indexed_obns
  const auto& parallel_attr =
      JUST(compatible_py::GetOpArgParallelAttribute(blob_parallel_desc_sym, op_attr, obn));
  const auto& blob_attr = JUST(compatible_py::GetOpArgBlobAttribute(op_attr, obn));

  CHECK_OR_RETURN(!outputs->at(0).get());
  (*outputs)[0] = JUST(OpInterpUtil::BuildTensor(blob_attr, parallel_attr, /* is_lazy= */ true,
                                                 /* is_local= */ input_tensor->is_local()));
  TensorNameScope::Global()->Record(outputs->at(0), GenLogicalBlobName(op_conf.name(), obn));
  return Maybe<void>::Ok();
}

Maybe<void> LazyInterpreter::ApplyImpl(const FeedVariableOpExpr& op_expr, const TensorTuple& inputs,
                                       TensorTuple* outputs, const OpExprInterpContext& ctx) const {
  // NOTE(chengcheng): inputs[0] is the EagerTensor
  CHECK_EQ_OR_RETURN(inputs.size(), 1);
  CHECK_EQ_OR_RETURN(op_expr.input_size(), 1);
  const std::shared_ptr<Tensor>& input_tensor = inputs.at(0);
  CHECK_OR_RETURN(input_tensor->is_eager());

  std::shared_ptr<Scope> scope = JUST(NewScopeWithParallelDescByTensor(input_tensor));

  OperatorConf op_conf;
  op_conf.set_name(op_expr.op_name());  // construct by python nn.Graph
  op_conf.set_scope_symbol_id(JUST(scope->symbol_id()));
  op_conf.set_device_tag(GetDeviceTagOfTensor(input_tensor));
  // NOTE(chengcheng):
  //   We contruct VariableOpConf instead of FeedVariableOpConf because FeedVariableOpExpr JUST
  //   for getting input EagerTensor.
  VariableOpConf* var_conf = op_conf.mutable_variable_conf();
  var_conf->set_out("out");
  input_tensor->shape()->ToProto(var_conf->mutable_shape());
  var_conf->set_data_type(input_tensor->dtype());
  // NOTE(chengcheng): VariableOpConf initializer_conf is useless because variable is inited
  //   by EagerTensor.
  var_conf->mutable_initializer()->mutable_empty_conf();
  JUST(GenVariableOpConfParallelDistributionStringByTensor(var_conf, input_tensor));
  if (!input_tensor->requires_grad()) { var_conf->set_trainable(false); }
  if (input_tensor->requires_grad()) {
    double l2 = JUST(ctx.attrs.GetAttr<double>("l2"));
    var_conf->mutable_regularizer()->mutable_l1_l2_conf()->set_l2(l2);
  }

  auto infer_ctx = JUST(GetCurInferCtx());
  OpAttribute op_attr = *JUST(infer_ctx->AddAndInferConsistentOp(op_conf));

  VLOG(2) << "Lazy nn.Graph name " << infer_ctx->job().job_conf().job_name() << " add op : \n"
          << op_conf.DebugString() << std::endl;
  VLOG(3) << "Lazy nn.Graph name " << infer_ctx->job().job_conf().job_name()
          << " infer and and op attr : \n"
          << op_attr.DebugString() << std::endl;

  int64_t parallel_desc_sym_id = JUST(scope->GetParallelDescSymbolId(op_conf));
  const std::shared_ptr<ParallelDesc>& blob_parallel_desc_sym =
      JUST(GetSymbol<cfg::ParallelConf, ParallelDesc>(parallel_desc_sym_id));

  // Check outputs num and setup output tensor properties.
  CHECK_EQ_OR_RETURN(outputs->size(), 1);
  CHECK_EQ_OR_RETURN(op_expr.output_size(), 1);

  const std::string obn = "out";  // NOTE(chengcheng): obn is NOT op_expr.indexed_obns
  const auto& parallel_attr =
      JUST(compatible_py::GetOpArgParallelAttribute(blob_parallel_desc_sym, op_attr, obn));
  const auto& blob_attr = JUST(compatible_py::GetOpArgBlobAttribute(op_attr, obn));

  CHECK_OR_RETURN(!outputs->at(0).get());
  (*outputs)[0] = JUST(OpInterpUtil::BuildTensor(blob_attr, parallel_attr, /* is_lazy= */ true,
                                                 /* is_local */ input_tensor->is_local()));
  // NOTE(chengcheng): Record variable op output LazyTenosr
  TensorNameScope::Global()->Record(outputs->at(0), GenLogicalBlobName(op_conf.name(), obn));
  // NOTE(chengcheng): Record EagerTensor as variable tensor name
  TensorNameScope::Global()->Record(input_tensor, GenLogicalBlobName(op_conf.name(), obn));
  return Maybe<void>::Ok();
}

Maybe<void> LazyInterpreter::ApplyImpl(const FetchOutputOpExpr& op_expr, const TensorTuple& inputs,
                                       TensorTuple* outputs, const OpExprInterpContext& ctx) const {
  // NOTE(chengcheng): inputs[0] is the LazyTensor
  CHECK_EQ_OR_RETURN(inputs.size(), 1);
  CHECK_EQ_OR_RETURN(op_expr.input_size(), 1);
  const std::shared_ptr<Tensor>& input_tensor = inputs.at(0);
  CHECK_OR_RETURN(input_tensor->is_lazy());
  const std::string& input_lbn = TensorNameScope::Global()->Lookup(input_tensor);
  CHECK_OR_RETURN(!input_lbn.empty());  // lbn must exist.

  std::shared_ptr<Scope> scope = JUST(NewScopeWithParallelDescByTensor(input_tensor));

  OperatorConf op_conf;
  op_conf.set_name(op_expr.op_name());  // construct by python nn.Graph
  op_conf.set_scope_symbol_id(JUST(scope->symbol_id()));
  op_conf.set_device_tag(GetDeviceTagOfTensor(input_tensor));
  // NOTE(chengcheng):
  //   We contruct OutputOpConf instead of FetchOutputOpConf because FetchOutputOpExpr JUST
  //   for get nn.Graph output LazyTensor.
  OutputOpConf* output_conf = op_conf.mutable_output_conf();
  output_conf->set_in(input_lbn);
  output_conf->set_out("out");
  InterfaceBlobConf* blob_conf = output_conf->mutable_blob_conf();
  input_tensor->shape()->ToProto(blob_conf->mutable_shape());
  blob_conf->set_data_type(input_tensor->dtype());
  // NOTE(chengcheng): is_dynamic true has conflict in consistent lazy job even if world size 1.
  //     this flag will be removed in the future.
  // blob_conf->set_is_dynamic(GetIsDynamicOfTensor(input_tensor));
  blob_conf->set_is_dynamic(false);
  JUST(GenParallelDistributionByTensor(blob_conf->mutable_nd_sbp(), input_tensor));

  auto infer_ctx = JUST(GetCurInferCtx());
  OpAttribute op_attr = *JUST(infer_ctx->AddAndInferConsistentOp(op_conf));

  VLOG(2) << "Lazy nn.Graph name " << infer_ctx->job().job_conf().job_name() << " add op : \n"
          << op_conf.DebugString() << std::endl;
  VLOG(3) << "Lazy nn.Graph name " << infer_ctx->job().job_conf().job_name()
          << " infer and and op attr : \n"
          << op_attr.DebugString() << std::endl;

  int64_t parallel_desc_sym_id = JUST(scope->GetParallelDescSymbolId(op_conf));
  const std::shared_ptr<ParallelDesc>& blob_parallel_desc_sym =
      JUST(GetSymbol<cfg::ParallelConf, ParallelDesc>(parallel_desc_sym_id));

  // Check outputs num and setup output tensor properties.
  CHECK_EQ_OR_RETURN(outputs->size(), 1);
  CHECK_EQ_OR_RETURN(op_expr.output_size(), 1);

  const std::string obn = "out";  // NOTE(chengcheng): obn is NOT op_expr.indexed_obns
  const auto& parallel_attr =
      JUST(compatible_py::GetOpArgParallelAttribute(blob_parallel_desc_sym, op_attr, obn));
  const auto& blob_attr = JUST(compatible_py::GetOpArgBlobAttribute(op_attr, obn));

  CHECK_OR_RETURN(!outputs->at(0).get());
  // TODO(chengcheng): Build EagerLocalTensor if parllel attr is this rank.
  (*outputs)[0] = JUST(OpInterpUtil::BuildTensor(blob_attr, parallel_attr, /* is_lazy= */ false,
                                                 /* is_local= */ input_tensor->is_local()));
  return Maybe<void>::Ok();
}

namespace {

Maybe<void> LazyInterpreterApplyImplForSourceUserOpExpr(const UserOpExpr& op_expr,
                                                        TensorTuple* outputs,
                                                        const OpExprInterpContext& ctx) {
  bool is_local;
  std::shared_ptr<const ParallelDesc> parallel_desc;
  if (ctx.parallel_desc.has_value()) {  // NOTE(chengcheng): consistent
    CHECK_OR_RETURN(!ctx.device.has_value());
    parallel_desc = JUST(ctx.parallel_desc.value()).shared_from_symbol();
    is_local = false;
  } else {
    CHECK_OR_RETURN(ctx.device.has_value());  // NOTE(chengcheng): local
    CHECK_OR_RETURN(!ctx.nd_sbp.has_value());
    parallel_desc = JUST(ctx.device.value())->parallel_desc_ptr();
    is_local = true;
  }
  std::shared_ptr<cfg::ParallelConf> parallel_conf = std::make_shared<cfg::ParallelConf>();
  parallel_conf->InitFromProto(parallel_desc->parallel_conf());
  const auto& scope = JUST(NewScopeWithParallelConfAndCurScope(parallel_conf));
  auto op_conf = JUST(OpInterpUtil::GenBuiltinOpConf(op_expr, ctx.attrs));
  op_conf->set_scope_symbol_id(JUST(scope->symbol_id()));
  op_conf->set_device_tag(parallel_conf->device_tag());

  auto infer_ctx = JUST(GetCurInferCtx());
  // NOTE(chengcheng): MUST reset unique op name before InferCtx::AddOp
  const std::string new_op_name = *JUST(infer_ctx->NewUniqueOpNameByFunctionalOpConf(*op_conf));

  // NOTE(chengcheng): for UserOp, NOT only reset op_name, but also the output values.
  op_conf->set_name(new_op_name);
  for (auto& pair : *(op_conf->mutable_user_conf()->mutable_output())) {
    auto& list_s = pair.second;
    for (int i = 0; i < list_s.s_size(); ++i) {
      std::string old_lbn = list_s.s(i);
      LogicalBlobId old_lbi = GenLogicalBlobId(old_lbn);
      // NOTE(chengcheng): MUST change the old_lbn to new op name.
      std::string new_lbn = GenLogicalBlobName(new_op_name, old_lbi.blob_name());
      list_s.set_s(i, new_lbn);
    }
  }

  OpAttribute op_attr = *JUST(infer_ctx->AddAndInferConsistentOp(*op_conf));

  VLOG(2) << "Lazy nn.Graph name " << infer_ctx->job().job_conf().job_name() << " add op : \n"
          << op_conf->DebugString() << std::endl;
  VLOG(3) << "Lazy nn.Graph name " << infer_ctx->job().job_conf().job_name()
          << " infer and and op attr : \n"
          << op_attr.DebugString() << std::endl;

  int64_t parallel_desc_sym_id = JUST(scope->GetParallelDescSymbolId(*op_conf));
  const std::shared_ptr<ParallelDesc>& blob_parallel_desc_sym =
      JUST(GetSymbol<cfg::ParallelConf, ParallelDesc>(parallel_desc_sym_id));

  // Check outputs num and setup output tensor properties.
  CHECK_EQ_OR_RETURN(outputs->size(), op_expr.output_size());
  for (int i = 0; i < op_expr.output_size(); ++i) {
    const std::string& obn = op_expr.indexed_obns().at(i);
    const auto& parallel_attr =
        JUST(compatible_py::GetOpArgParallelAttribute(blob_parallel_desc_sym, op_attr, obn));
    const auto& blob_attr = JUST(compatible_py::GetOpArgBlobAttribute(op_attr, obn));
    CHECK_OR_RETURN(!outputs->at(i).get());
    (*outputs)[i] = JUST(OpInterpUtil::BuildTensor(blob_attr, parallel_attr,
                                                   /* is_lazy= */ true, is_local));
    TensorNameScope::Global()->Record(outputs->at(i), GenLogicalBlobName(new_op_name, obn));
  }
  return Maybe<void>::Ok();
}

Maybe<void> AddFreeEagerTensorToVariableOp(const std::shared_ptr<Tensor>& input_tensor) {
  CHECK_OR_RETURN(input_tensor->is_eager());
  const std::string& empty_lbn = TensorNameScope::Global()->Lookup(input_tensor);
  CHECK_OR_RETURN(empty_lbn.empty());
  std::shared_ptr<Scope> scope = JUST(NewScopeWithParallelDescByTensor(input_tensor));
  OperatorConf op_conf;
  op_conf.set_scope_symbol_id(JUST(scope->symbol_id()));
  op_conf.set_device_tag(GetDeviceTagOfTensor(input_tensor));
  VariableOpConf* var_conf = op_conf.mutable_variable_conf();
  var_conf->set_out("out");
  input_tensor->shape()->ToProto(var_conf->mutable_shape());
  var_conf->set_data_type(input_tensor->dtype());
  // NOTE(chengcheng): VariableOpConf initializer_conf is useless because variable is inited
  //   by EagerTensor.
  var_conf->mutable_initializer()->mutable_empty_conf();
  JUST(GenVariableOpConfParallelDistributionStringByTensor(var_conf, input_tensor));
  // NOTE(chengcheng): Free EagerTensor not trainable
  var_conf->set_trainable(false);

  auto infer_ctx = JUST(GetCurInferCtx());
  // NOTE(chengcheng): MUST reset unique op name before InferCtx::AddOp, FreeEagerTensor has no
  //  name so just new a unique name for it.
  const std::string new_op_name = *JUST(infer_ctx->NewUniqueOpNameByFunctionalOpConf(op_conf));
  op_conf.set_name(new_op_name);

  OpAttribute op_attr = *JUST(infer_ctx->AddAndInferConsistentOp(op_conf));

  VLOG(2) << "Lazy nn.Graph name " << infer_ctx->job().job_conf().job_name() << " add op : \n"
          << op_conf.DebugString() << " for FreeEagerTensor.\n";
  VLOG(3) << "Lazy nn.Graph name " << infer_ctx->job().job_conf().job_name()
          << " infer and and op attr : \n"
          << op_attr.DebugString() << " for FreeEagerTensor.\n";

  // NOTE(chengcheng): MUST store this tensor to MultiClientSessionContext for graph runtime bind.
  const std::string graph_name = *JUST(JUST(GlobalJobBuildAndInferCtxMgr())->GetCurrentJobName());
  const std::string lbn = GenLogicalBlobName(new_op_name, "out");
  Global<MultiClientSessionContext>::Get()->StoreFreeEagerTensorWithNameByGraphName(
      graph_name, input_tensor, lbn);
  // NOTE(chengcheng): MUST record this eager_tensor name as new variable output lbn.
  TensorNameScope::Global()->Record(input_tensor, lbn);

  return Maybe<void>::Ok();
}

Maybe<void> LazyInterpreterApplyImplForCopyUserOpExpr(const UserOpExpr& op_expr,
                                                      const TensorTuple& inputs,
                                                      TensorTuple* outputs,
                                                      const OpExprInterpContext& ctx) {
  CHECK_OR_RETURN(op_expr.op_type_name() == "copy");
  CHECK_EQ_OR_RETURN(inputs.size(), 1);
  CHECK_EQ_OR_RETURN(op_expr.input_size(), 1);
  const std::shared_ptr<Tensor>& input_tensor = inputs.at(0);
  CHECK_OR_RETURN(input_tensor->is_lazy());
  std::string input_lbn = TensorNameScope::Global()->Lookup(input_tensor);
  if (input_lbn.empty()) {
    JUST(AddFreeEagerTensorToVariableOp(input_tensor));
    input_lbn = TensorNameScope::Global()->Lookup(input_tensor);
  }
  CHECK_OR_RETURN(!input_lbn.empty());  // lbn must exist.
  std::string device_type = JUST(ctx.attrs.GetAttr<std::string>("device_type"));
  int64_t device_id = JUST(ctx.attrs.GetAttr<int64_t>("device_id"));

  CHECK_EQ_OR_RETURN(outputs->size(), 1);
  CHECK_EQ_OR_RETURN(op_expr.output_size(), 1);
  if (input_tensor->is_local()) {
    (*outputs)[0] = JUST(MirroredTensor::MakeTensor(input_tensor->shape(), input_tensor->dtype(),
                                                    JUST(Device::New(device_type, device_id)),
                                                    /* is_lazy= */ true,
                                                    /*requires_grad=*/false, /*is_leaf=*/true));
  } else {
    ParallelConf parallel_conf = JUST(input_tensor->parallel_desc())->parallel_conf();
    parallel_conf.set_device_tag(GetDeviceTagByDeviceTypeStr(device_type));
    ParallelDesc parallel_desc(parallel_conf);
    (*outputs)[0] =
        JUST(ConsistentTensor::MakeTensor(input_tensor->shape(), input_tensor->dtype(),
                                          JUST(input_tensor->nd_sbp()), SymbolOf(parallel_desc),
                                          /* is_lazy= */ true,
                                          /*requires_grad=*/false, /*is_leaf=*/true));
  }
  // NOTE(chengcheng): output tensor lbn is SAME with input tensor.
  TensorNameScope::Global()->Record(outputs->at(0), input_lbn);
  return Maybe<void>::Ok();
}

}  // namespace

Maybe<void> LazyInterpreter::ApplyImpl(const UserOpExpr& op_expr, const TensorTuple& inputs,
                                       TensorTuple* outputs, const OpExprInterpContext& ctx) const {
  CHECK_EQ_OR_RETURN(inputs.size(), op_expr.input_size());

  // NOTE(chengcheng): Handle special UserOp such as:
  //     1. [Source UserOp] : OFRecordReader, CoinFlip
  //     2. [Change Placement/ParallelDesc UserOp] : to(copy)/to_consistent/parallel_cast
  //     3. [Multi-Inputs & Different ParallelDesc for each input UserOp] : like there are 2 inputs,
  //             one from CPU and the other from GPU.
  //     ..., etc.
  //
  //     Need add if for each special UserOp for infer:
  //     1. op_conf: device_tag,
  //     2. output tensor: is_local,
  //     3. op_parallel_conf for build new scope with parallel_desc
  //     4. output blob (different with tensor) -> parallel_conf
  //     5. need add to JobBuildAndInferCtx (like copy will NOT need)
  if (inputs.size() == 0) {
    // NOTE(chengcheng): handle for source UserOp like OFRecordReader, CoinFlip
    return LazyInterpreterApplyImplForSourceUserOpExpr(op_expr, outputs, ctx);
  }
  if (op_expr.op_type_name() == "copy") {
    // NOTE(chengcheng): handle for copy UserOp which will NOT add op to job.
    return LazyInterpreterApplyImplForCopyUserOpExpr(op_expr, inputs, outputs, ctx);
  }

  // NOTE(chengcheng):
  //   Normal UserOp inputs size >= 1 for infer parallel_desc.
  CHECK_GE_OR_RETURN(inputs.size(), 1);
  auto op_conf = JUST(OpInterpUtil::GenBuiltinOpConf(op_expr, ctx.attrs));
  std::shared_ptr<Scope> scope = JUST(NewScopeWithParallelDescByTensor(inputs.at(0)));
  op_conf->set_scope_symbol_id(JUST(scope->symbol_id()));
  const std::string device_tag = GetDeviceTagOfTensor(inputs.at(0));
  const bool is_local = inputs.at(0)->is_local();
  const std::shared_ptr<const ParallelDesc> parallel_desc =
      JUST(GetParallelDescOfTensor(inputs.at(0)));

  op_conf->set_device_tag(device_tag);
  for (int i = 0; i < inputs.size(); ++i) {
    const auto& input_tensor = inputs.at(i);
    CHECK_OR_RETURN(device_tag == GetDeviceTagOfTensor(input_tensor));
    CHECK_OR_RETURN(
        parallel_desc->EqualsIgnoringHierarchy(*JUST(GetParallelDescOfTensor(input_tensor))));
    CHECK_EQ_OR_RETURN(is_local, input_tensor->is_local());
    const std::string& ibn = op_expr.indexed_ibns().at(i);
    std::string lbn = TensorNameScope::Global()->Lookup(input_tensor);
    if (lbn.empty()) {
      JUST(AddFreeEagerTensorToVariableOp(input_tensor));
      lbn = TensorNameScope::Global()->Lookup(input_tensor);
    }
    CHECK_OR_RETURN(!lbn.empty());  // NOTE(chengcheng): lbn must not empty now.
    ReplaceInputLbnInOpCustomizedConf(op_conf.get(), ibn, lbn);
  }

  auto infer_ctx = JUST(GetCurInferCtx());
  // NOTE(chengcheng): MUST reset unique op name before InferCtx::AddOp
  const std::string new_op_name = *JUST(infer_ctx->NewUniqueOpNameByFunctionalOpConf(*op_conf));

  // NOTE(chengcheng): for UserOp, NOT only reset op_name, but also the output values.
  op_conf->set_name(new_op_name);
  for (auto& pair : *(op_conf->mutable_user_conf()->mutable_output())) {
    auto& list_s = pair.second;
    for (int i = 0; i < list_s.s_size(); ++i) {
      std::string old_lbn = list_s.s(i);
      LogicalBlobId old_lbi = GenLogicalBlobId(old_lbn);
      // NOTE(chengcheng): MUST change the old_lbn to new op name.
      std::string new_lbn = GenLogicalBlobName(new_op_name, old_lbi.blob_name());
      list_s.set_s(i, new_lbn);
    }
  }

  OpAttribute op_attr = *JUST(infer_ctx->AddAndInferConsistentOp(*op_conf));

  VLOG(2) << "Lazy nn.Graph name " << infer_ctx->job().job_conf().job_name() << " add op : \n"
          << op_conf->DebugString() << std::endl;
  VLOG(3) << "Lazy nn.Graph name " << infer_ctx->job().job_conf().job_name()
          << " infer and and op attr : \n"
          << op_attr.DebugString() << std::endl;

  int64_t parallel_desc_sym_id = JUST(scope->GetParallelDescSymbolId(*op_conf));
  const std::shared_ptr<ParallelDesc>& blob_parallel_desc_sym =
      JUST(GetSymbol<cfg::ParallelConf, ParallelDesc>(parallel_desc_sym_id));

  // Check outputs num and setup output tensor properties.
  CHECK_EQ_OR_RETURN(outputs->size(), op_expr.output_size());
  for (int i = 0; i < op_expr.output_size(); ++i) {
    const std::string& obn = op_expr.indexed_obns().at(i);
    const auto& parallel_attr =
        JUST(compatible_py::GetOpArgParallelAttribute(blob_parallel_desc_sym, op_attr, obn));
    const auto& blob_attr = JUST(compatible_py::GetOpArgBlobAttribute(op_attr, obn));
    if (!(outputs->at(i).get())) {
      (*outputs)[i] = JUST(OpInterpUtil::BuildTensor(blob_attr, parallel_attr,
                                                     /* is_lazy= */ true, is_local));
    } else {
      const std::shared_ptr<Tensor>& inplace_out = outputs->at(i);
      JUST(OpInterpUtil::CheckTensorMatchAttr(inplace_out, blob_attr, parallel_attr,
                                              /* is_lazy= */ true, is_local,
                                              /* requires_grad */ false,
                                              /* is_leaf */ true));
    }
    TensorNameScope::Global()->Record(outputs->at(i), GenLogicalBlobName(new_op_name, obn));
  }
  return Maybe<void>::Ok();
}

Maybe<void> LazyInterpreter::ApplyImpl(const FunctionOpExpr& op_expr, const TensorTuple& inputs,
                                       TensorTuple* outputs, const OpExprInterpContext& ctx) const {
  // TODO(hjchen2)
  OF_UNIMPLEMENTED() << "The type " << op_expr.op_type_name()
                     << " has not been supported in LazyInterpreter::Apply.";
  return Maybe<void>::Ok();
}

}  // namespace one
}  // namespace oneflow
