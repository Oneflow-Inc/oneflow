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
#include "oneflow/core/framework/op_interpreter/lazy_op_interpreter.h"

#include <memory>
#include "oneflow/core/common/cpp_attribute.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/cpp_attribute.h"
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/framework/consistency_check.h"
#include "oneflow/core/framework/user_op_conf.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/multi_client_session_context.h"
#include "oneflow/core/framework/op_interpreter.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/framework/scope_util.h"
#include "oneflow/core/framework/session_util.h"
#include "oneflow/core/framework/symbol_storage_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_name_scope.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/framework/user_op_registry.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/lazy_mode.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/job/sbp_parallel.h"
#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"
#include "oneflow/core/vm/vm_util.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

namespace {

Maybe<Tensor> BuildTensor(const OpAttribute& op_attribute, const std::string& bn_in_op,
                          const std::shared_ptr<ParallelDesc>& parallel_desc, const bool is_lazy,
                          const bool is_local) {
  CHECK_OR_RETURN(op_attribute.has_logical_blob_desc_signature());  // NOLINT(maybe-need-error-msg)
  const auto& blob_desc_sign_map = op_attribute.logical_blob_desc_signature().bn_in_op2blob_desc();
  auto blob_desc_it = blob_desc_sign_map.find(bn_in_op);
  CHECK_OR_RETURN(blob_desc_it != blob_desc_sign_map.end())
      << "blob_desc of " << bn_in_op << " not found in op " << op_attribute.op_conf().name();

  auto shape = std::make_shared<Shape>(blob_desc_it->second.shape());
  auto stride = std::make_shared<Stride>(shape);
  auto dtype = blob_desc_it->second.data_type();
  if (is_local) {
    const auto& device = JUST(Device::MakeDeviceByParallelDesc(*parallel_desc));
    const auto& tensor =
        JUST(LocalTensor::MakeTensor(shape, stride, dtype, device, is_lazy,
                                     /* requires_grad= */ false, /* is_leaf= */ true));
    return static_cast<std::shared_ptr<Tensor>>(tensor);
  } else {
    const auto& nd_sbp_sign_map = op_attribute.nd_sbp_signature().bn_in_op2nd_sbp();
    auto nd_sbp_it = nd_sbp_sign_map.find(bn_in_op);
    CHECK_OR_RETURN(nd_sbp_it != nd_sbp_sign_map.end())
        << "nd_sbp of " << bn_in_op << " not found in op " << op_attribute.op_conf().name();
    NdSbp nd_sbp(nd_sbp_it->second);
    const auto& tensor = JUST(GlobalTensor::MakeTensor(shape, dtype, SymbolOf(nd_sbp),
                                                       SymbolOf(*parallel_desc), is_lazy,
                                                       /*requires_grad=*/false, /*is_leaf=*/true));
    return static_cast<std::shared_ptr<Tensor>>(tensor);
  }
}

Maybe<void> CheckTensorMatchAttr(const std::shared_ptr<Tensor>& tensor,
                                 const OpAttribute& op_attribute, const std::string& bn_in_op,
                                 const std::shared_ptr<ParallelDesc>& parallel_desc,
                                 const bool is_local) {
  CHECK_EQ_OR_RETURN(tensor->is_local(), is_local);  // NOLINT(maybe-need-error-msg)

  CHECK_OR_RETURN(op_attribute.has_logical_blob_desc_signature());  // NOLINT(maybe-need-error-msg)
  const auto& blob_desc_sign_map = op_attribute.logical_blob_desc_signature().bn_in_op2blob_desc();
  auto blob_desc_it = blob_desc_sign_map.find(bn_in_op);
  CHECK_OR_RETURN(blob_desc_it != blob_desc_sign_map.end())
      << "blob_desc of " << bn_in_op << " not found in op " << op_attribute.op_conf().name();

  auto shape = std::make_shared<Shape>(blob_desc_it->second.shape());
  auto dtype = blob_desc_it->second.data_type();
  CHECK_EQ_OR_RETURN(*tensor->shape(), *shape);             // NOLINT(maybe-need-error-msg)
  CHECK_EQ_OR_RETURN(tensor->dtype()->data_type(), dtype);  // NOLINT(maybe-need-error-msg)

  if (is_local) {
    const auto& device = JUST(Device::MakeDeviceByParallelDesc(*parallel_desc));
    CHECK_OR_RETURN(JUST(tensor->device()) == device);  // NOLINT(maybe-need-error-msg)
  } else {
    const auto& nd_sbp_sign_map = op_attribute.nd_sbp_signature().bn_in_op2nd_sbp();
    auto nd_sbp_it = nd_sbp_sign_map.find(bn_in_op);
    CHECK_OR_RETURN(nd_sbp_it != nd_sbp_sign_map.end())
        << "nd_sbp of " << bn_in_op << " not found in op " << op_attribute.op_conf().name();
    // Only check the nd_sbp if auto parallel is not enable,
    // since the semi-auto parallellism rule might have inconsistency with the auto-parallel
    // strategy.
    if (!GlobalJobDesc().enable_auto_parallel()) {
      NdSbp nd_sbp(nd_sbp_it->second);
      CHECK_OR_RETURN(JUST(tensor->nd_sbp()) == SymbolOf(nd_sbp))
          << "The input sbp is not valid for an inplace operation, please try to use non-inplace. "
          << NdSbpToString(JUST(tensor->nd_sbp())) << " vs " << NdSbpToString(nd_sbp);
    }
    CHECK_OR_RETURN(JUST(tensor->parallel_desc())  // NOLINT(maybe-need-error-msg)
                    == SymbolOf(*parallel_desc));  // NOLINT(maybe-need-error-msg)
  }
  return Maybe<void>::Ok();
}

Maybe<const std::string&> GetDeviceTagOfTensor(const std::shared_ptr<Tensor>& tensor) {
  if (tensor->is_global()) { return JUST(tensor->parallel_desc())->device_tag(); }
  return JUST(tensor->device())->type();
}

bool GetIsDynamicOfTensor(const std::shared_ptr<Tensor>& tensor) {
  if (tensor->is_global()) {
    return false;
  } else {
    return true;
  }
}

Maybe<void> GenNdSbpByTensor(NdSbp* nd_sbp, const std::shared_ptr<Tensor>& tensor) {
  nd_sbp->clear_sbp_parallel();
  if (tensor->is_local()) {
    // NOTE(chengcheng):
    //   OneFlow Lazy is always global. LocalTensor is a special case of GlobalTensor
    //   which placement is only this rank, and SbpParallel is Broadcast.
    nd_sbp->add_sbp_parallel()->mutable_broadcast_parallel();
  } else {
    *nd_sbp = *JUST(tensor->nd_sbp());
  }
  return Maybe<void>::Ok();
}

Maybe<void> GenVariableOpConfNdSbpStringByTensor(VariableOpConf* var_conf,
                                                 const std::shared_ptr<Tensor>& tensor) {
  var_conf->clear_nd_sbp();
  if (tensor->is_local()) {
    SbpParallel broadcast;
    broadcast.mutable_broadcast_parallel();
    var_conf->add_nd_sbp(SbpParallelToString(broadcast));
  } else {
    const NdSbp& nd_sbp = *JUST(tensor->nd_sbp());
    for (const auto& sbp_parallel : nd_sbp.sbp_parallel()) {
      var_conf->add_nd_sbp(SbpParallelToString(sbp_parallel));
    }
  }
  return Maybe<void>::Ok();
}

Maybe<const ParallelDesc> GetParallelDescOfTensor(const std::shared_ptr<Tensor>& tensor) {
  if (tensor->is_local()) {
    const auto& device = JUST(tensor->device());
    const auto& placement = JUST(Placement4Device(device));
    return placement.shared_from_symbol();
  } else {
    return JUST(tensor->parallel_desc()).shared_from_symbol();
  }
}

Maybe<Scope> NewScopeWithParallelConfAndCurScope(const ParallelConf& parallel_conf) {
  std::shared_ptr<Scope> new_scope;
  const auto& old_scope = JUST(GetCurrentScope());
  JUST(PhysicalRun([&](InstructionsBuilder* builder) -> Maybe<void> {
    new_scope = JUST(builder->BuildScopeWithNewParallelConf(old_scope, parallel_conf));
    return Maybe<void>::Ok();
  }));
  // NOTE(chengcheng): need sync vm for get scope right now
  JUST(vm::CurrentRankSync());
  CHECK_OR_RETURN(new_scope);  // NOLINT(maybe-need-error-msg)
  return new_scope;
}

Maybe<Scope> NewScopeWithParallelDescByTensor(const std::shared_ptr<Tensor>& tensor) {
  return NewScopeWithParallelConfAndCurScope(
      JUST(GetParallelDescOfTensor(tensor))->parallel_conf());
}

Maybe<int32_t> GetGradAccStep() {
  const auto& infer_ctx = JUST(GetCurInferCtx());
  const auto& job_conf = infer_ctx->job().job_conf();
  if (job_conf.has_train_conf() && job_conf.has_num_gradient_accumulation_steps()
      && job_conf.num_gradient_accumulation_steps() > 1) {
    return job_conf.num_gradient_accumulation_steps();
  } else {
    return 1;
  }
}

Maybe<void> AddFreeEagerTensorToVariableOp(const std::shared_ptr<Tensor>& input_tensor) {
  if (!input_tensor->is_contiguous()) {
    LazyMode::Guard lazy_mode_disabled_guard(false);
    JUST(functional::InplaceToContiguous(input_tensor));
    JUST(vm::CurrentRankSync());
  }

  CHECK_OR_RETURN(input_tensor->is_eager());  // NOLINT(maybe-need-error-msg)
  const std::string& empty_lbn = TensorNameScope::Global()->Lookup(input_tensor);
  CHECK_OR_RETURN(empty_lbn.empty());  // NOLINT(maybe-need-error-msg)
  std::shared_ptr<Scope> scope = JUST(NewScopeWithParallelDescByTensor(input_tensor));
  OperatorConf op_conf;
  op_conf.set_scope_symbol_id(JUST(scope->symbol_id()));
  op_conf.set_device_tag(JUST(GetDeviceTagOfTensor(input_tensor)));
  VariableOpConf* var_conf = op_conf.mutable_variable_conf();
  var_conf->set_out("out");
  input_tensor->shape()->ToProto(var_conf->mutable_shape());
  var_conf->set_data_type(input_tensor->dtype()->data_type());
  // NOTE(chengcheng): VariableOpConf initializer_conf is useless because variable is inited
  //   by EagerTensor.
  var_conf->mutable_initializer()->mutable_empty_conf();
  JUST(GenVariableOpConfNdSbpStringByTensor(var_conf, input_tensor));
  // NOTE(chengcheng): Free EagerTensor not trainable
  var_conf->set_trainable(false);

  auto infer_ctx = JUST(GetCurInferCtx());
  // NOTE(chengcheng): MUST reset unique op name before InferCtx::AddOp, FreeEagerTensor has no
  //  name so just new a unique name for it.
  const std::string new_op_name = *JUST(infer_ctx->NewUniqueOpNameByFunctionalOpConf(op_conf));
  op_conf.set_name(new_op_name);

  VLOG(2) << "Lazy nn.Graph name " << infer_ctx->job().job_conf().job_name() << " try to add op: \n"
          << op_conf.DebugString() << std::endl;
  OpAttribute op_attr = *JUST(infer_ctx->AddAndInferGlobalOp(op_conf));
  VLOG(2) << "Lazy nn.Graph name " << infer_ctx->job().job_conf().job_name() << " add op : \n"
          << op_conf.name() << " for FreeEagerTensor.\n";
  VLOG(3) << "Lazy nn.Graph name " << infer_ctx->job().job_conf().job_name()
          << " infer and and op attr : \n"
          << op_attr.DebugString() << " for FreeEagerTensor.\n";

  // NOTE(chengcheng): MUST store this tensor to MultiClientSessionContext for graph runtime bind.
  const std::string graph_name = *JUST(JUST(GlobalJobBuildAndInferCtxMgr())->GetCurrentJobName());
  const std::string lbn = GenLogicalBlobName(new_op_name, "out");
  Singleton<MultiClientSessionContext>::Get()->StoreFreeEagerTensorWithNameByGraphName(
      graph_name, input_tensor, new_op_name);

  int64_t parallel_desc_sym_id = JUST(scope->GetParallelDescSymbolId(op_conf));
  auto blob_parallel_desc = JUST(GetSymbol<ParallelDesc>(parallel_desc_sym_id));

  auto var_tensor = JUST(BuildTensor(op_attr, "out", blob_parallel_desc, /* is_lazy= */ true,
                                     /* is_local= */ input_tensor->is_local()));
  TensorNameScope::Global()->Record(var_tensor, lbn);

  // NOTE(chengcheng): MUST record this eager_tensor name as new variable output lbn.
  // NOTE(chengcheng): in GradAcc FreeEagerTensor need insert repeat op, but there is no need to
  //  create a new tensor for repeat op out. We just set repeat lbn as this free eager tensor's lbn.
  auto repeat_tensor = JUST(GradAccTryInsertRepeatAfterVar(var_tensor));
  const std::string& repeat_tensor_name = TensorNameScope::Global()->Lookup(repeat_tensor);
  CHECK_OR_RETURN(!repeat_tensor_name.empty());  // NOLINT(maybe-need-error-msg)
  TensorNameScope::Global()->Record(input_tensor, repeat_tensor_name);
  return Maybe<void>::Ok();
}

}  // namespace

Maybe<Tensor> GradAccTryInsertUnpackAfterInput(const std::shared_ptr<Tensor>& input) {
  int32_t grad_acc_step = JUST(GetGradAccStep());
  if (grad_acc_step > 1) {
    // NOTE(chengcheng):
    //   We assume that the input data is one mini-batch which containing multi micro-batches.
    //   So we need unpack input data for each micro-batch.
    VLOG(2)
        << " Current OneFlow nn.Graph grad acc semantics is different from Torch. \n"
        << " Once call nn.Graph in OneFlow, it indicates a mini-batch. When grad acc steps > 1, \n"
        << " the input tensor of nn.Graph will be unpacked by 0th dim into multiple micro-batches "
        << " and exec them in order.\n";
    const auto& infer_ctx = JUST(GetCurInferCtx());
    const auto& input_lbn = TensorNameScope::Global()->Lookup(input);
    VLOG(2) << "Lazy nn.Graph name " << infer_ctx->job().job_conf().job_name()
            << " add grad acc unpack op after input " << input_lbn << std::endl;
    return functional::GradAccUnpack(input, grad_acc_step);
  } else {
    return input;
  }
}

Maybe<Tensor> GradAccTryInsertRepeatAfterVar(const std::shared_ptr<Tensor>& variable) {
  int32_t grad_acc_step = JUST(GetGradAccStep());
  if (grad_acc_step > 1) {
    // NOTE(chengcheng):
    //   We assume that the nn.Graph once call is one mini-batch which containing multi
    //   micro-batches. So we just repeat variable tensor for each micro-batch.
    VLOG(2)
        << " Current OneFlow nn.Graph grad acc semantics is different from Torch. \n"
        << " Once call nn.Graph in OneFlow, it indicates a mini-batch. When grad acc steps > 1, \n"
        << " the var tensor of nn.Graph will be repeated exec for multiple micro-batches. \n";
    const auto& infer_ctx = JUST(GetCurInferCtx());
    const auto& variable_lbn = TensorNameScope::Global()->Lookup(variable);
    VLOG(2) << "Lazy nn.Graph name " << infer_ctx->job().job_conf().job_name()
            << " add grad acc repeat op after variable " << variable_lbn << std::endl;
    return functional::GradAccRepeat(variable, grad_acc_step);
  } else {
    return variable;
  }
}

Maybe<Tensor> GradAccTryInsertPackBeforeOutput(const std::shared_ptr<Tensor>& output) {
  int32_t grad_acc_step = JUST(GetGradAccStep());
  if (grad_acc_step > 1) {
    // NOTE(chengcheng):
    //   We assume that the nn.Graph once call is one mini-batch which containing multi
    //   micro-batches. So we need pack output tensor for each micro-batch to one micro-batch.
    VLOG(2)
        << " Current OneFlow nn.Graph grad acc semantics is different from Torch. \n"
        << " Once call nn.Graph in OneFlow, it indicates a mini-batch. When grad acc steps > 1, \n"
        << " the output tensor of nn.Graph will be packed to a big tensor by 0th dim, after exec \n"
        << " for multiple micro-batches. \n";
    const auto& infer_ctx = JUST(GetCurInferCtx());
    const auto& output_lbn = TensorNameScope::Global()->Lookup(output);
    VLOG(2) << "Lazy nn.Graph name " << infer_ctx->job().job_conf().job_name()
            << " add grad acc pack op before output " << output_lbn << std::endl;
    return functional::GradAccPack(output, grad_acc_step);
  } else {
    return output;
  }
}

Maybe<void> GradAccTryInsertRepeatTickBeforeSource(
    const std::shared_ptr<OperatorConf>& source_op_conf, bool is_local) {
  int32_t grad_acc_step = JUST(GetGradAccStep());
  if (grad_acc_step > 1) {
    // NOTE(chengcheng):
    //   We assume that the nn.Graph once call is one mini-batch which containing multi
    //   micro-batches. So we need repeat source op for each micro-batch in one micro-batch.
    VLOG(2)
        << " Current OneFlow nn.Graph grad acc semantics is different from Torch. \n"
        << " Once call nn.Graph in OneFlow, it indicates a mini-batch. When grad acc steps > 1, \n"
        << " the source op of nn.Graph will be repeated exec n-times for multiple micro-batches.\n";
    const auto& infer_ctx = JUST(GetCurInferCtx());
    // Insert Tick
    OperatorConf tick_conf{};
    tick_conf.set_name("Sys-GradAcc-RepeatTick-DeviceTick-" + source_op_conf->name());
    tick_conf.set_device_tag(source_op_conf->device_tag());
    tick_conf.mutable_device_tick_conf()->set_out("out");
    tick_conf.set_scope_symbol_id(source_op_conf->scope_symbol_id());
    auto tick_lbn = GenLogicalBlobName(tick_conf.name(), tick_conf.device_tick_conf().out());
    OpAttribute tick_op_attr = *JUST(infer_ctx->AddAndInferGlobalOp(tick_conf));
    VLOG(2) << "Lazy nn.Graph name " << infer_ctx->job().job_conf().job_name() << " add op: \n"
            << tick_conf.DebugString() << std::endl;
    VLOG(3) << "Lazy nn.Graph name " << infer_ctx->job().job_conf().job_name()
            << " infer and and op attr : \n"
            << tick_op_attr.DebugString() << std::endl;

    const auto& scope =
        Singleton<symbol::Storage<Scope>>::Get()->Get(source_op_conf->scope_symbol_id());
    int64_t parallel_desc_sym_id = JUST(scope.GetParallelDescSymbolId(tick_conf));
    auto blob_parallel_desc = JUST(GetSymbol<ParallelDesc>(parallel_desc_sym_id));

    auto tick_tensor = JUST(BuildTensor(tick_op_attr, tick_conf.device_tick_conf().out(),
                                        blob_parallel_desc, /* is_lazy= */ true,
                                        /* is_local= */ is_local));
    TensorNameScope::Global()->Record(tick_tensor, tick_lbn);

    VLOG(2) << "Lazy nn.Graph name " << infer_ctx->job().job_conf().job_name()
            << " add grad acc repeat op after tick op " << tick_conf.name()
            << " and before source op" << source_op_conf->name();
    auto repeat_tensor = JUST(functional::GradAccRepeat(tick_tensor, grad_acc_step));
    const std::string& repeat_tensor_name = TensorNameScope::Global()->Lookup(repeat_tensor);
    CHECK_OR_RETURN(!repeat_tensor_name.empty());  // NOLINT(maybe-need-error-msg)
    (*source_op_conf->mutable_user_conf()->mutable_input())[user_op::kUserSourceOpTickInputArgName]
        .add_s(repeat_tensor_name);
  }
  return Maybe<void>::Ok();
}

Maybe<void> LazyInterpreter::ApplyImpl(const FeedInputOpExpr& op_expr, const TensorTuple& inputs,
                                       TensorTuple* outputs, const OpExprInterpContext& ctx) const {
  // NOTE(chengcheng): inputs[0] is the EagerTensor
  CHECK_EQ_OR_RETURN(inputs.size(), 1);         // NOLINT(maybe-need-error-msg)
  CHECK_EQ_OR_RETURN(op_expr.input_size(), 1);  // NOLINT(maybe-need-error-msg)
  const std::shared_ptr<Tensor>& input_tensor = inputs.at(0);
  CHECK_OR_RETURN(input_tensor->is_eager());  // NOLINT(maybe-need-error-msg)

  std::shared_ptr<Scope> scope = JUST(NewScopeWithParallelDescByTensor(input_tensor));

  OperatorConf op_conf;
  op_conf.set_name(op_expr.op_name());  // construct by python nn.Graph
  op_conf.set_scope_symbol_id(JUST(scope->symbol_id()));
  op_conf.set_device_tag(JUST(GetDeviceTagOfTensor(input_tensor)));
  // NOTE(chengcheng):
  //   We contruct InputOpConf instead of FeedInputOpConf because FeedInputOpExpr JUST for getting
  //   input EagerTensor.
  InputOpConf* input_conf = op_conf.mutable_input_conf();
  input_conf->set_out("out");
  InterfaceBlobConf* blob_conf = input_conf->mutable_blob_conf();

  input_tensor->shape()->ToProto(blob_conf->mutable_shape());
  blob_conf->set_data_type(input_tensor->dtype()->data_type());
  // NOTE(chengcheng): is_dynamic true has conflict in global lazy job even if world size 1.
  //     this flag will be removed in the future.
  // blob_conf->set_is_dynamic(GetIsDynamicOfTensor(input_tensor));
  blob_conf->set_is_dynamic(false);
  JUST(GenNdSbpByTensor(blob_conf->mutable_nd_sbp(), input_tensor));

  auto infer_ctx = JUST(GetCurInferCtx());
  VLOG(2) << "Lazy nn.Graph name " << infer_ctx->job().job_conf().job_name()
          << " try to add op: \n: " << op_conf.DebugString() << std::endl;
  OpAttribute op_attr = *JUST(infer_ctx->AddAndInferGlobalOp(op_conf));
  VLOG(2) << "Lazy nn.Graph name " << infer_ctx->job().job_conf().job_name() << " add op : \n"
          << op_conf.name() << std::endl;
  VLOG(3) << "Lazy nn.Graph name " << infer_ctx->job().job_conf().job_name()
          << " infer and and op attr : \n"
          << op_attr.DebugString() << std::endl;

  int64_t parallel_desc_sym_id = JUST(scope->GetParallelDescSymbolId(op_conf));
  auto blob_parallel_desc = JUST(GetSymbol<ParallelDesc>(parallel_desc_sym_id));

  // Check outputs num and setup output tensor properties.
  CHECK_EQ_OR_RETURN(outputs->size(), 1);        // NOLINT(maybe-need-error-msg)
  CHECK_EQ_OR_RETURN(op_expr.output_size(), 1);  // NOLINT(maybe-need-error-msg)
  CHECK_OR_RETURN(!(*outputs)[0]);               // NOLINT(maybe-need-error-msg)
  const std::string obn = "out";  // NOTE(chengcheng): obn is NOT op_expr.indexed_obns
  auto origin_input = JUST(BuildTensor(op_attr, obn, blob_parallel_desc, /* is_lazy= */ true,
                                       /* is_local= */ input_tensor->is_local()));
  TensorNameScope::Global()->Record(origin_input, GenLogicalBlobName(op_conf.name(), obn));
  TensorNameScope::Global()->Record(input_tensor, GenLogicalBlobName(op_conf.name(), obn));

  // NOTE: The input will then be unpacked in DispatchFeedInputOpExprFunctor
  // if GradAcc is enabled
  (*outputs)[0] = origin_input;
  return Maybe<void>::Ok();
}

Maybe<void> LazyInterpreter::ApplyImpl(const FeedVariableOpExpr& op_expr, const TensorTuple& inputs,
                                       TensorTuple* outputs, const OpExprInterpContext& ctx) const {
  // NOTE(chengcheng): inputs[0] is the EagerTensor
  CHECK_EQ_OR_RETURN(inputs.size(), 1);         // NOLINT(maybe-need-error-msg)
  CHECK_EQ_OR_RETURN(op_expr.input_size(), 1);  // NOLINT(maybe-need-error-msg)
  const std::shared_ptr<Tensor>& input_tensor = inputs.at(0);
  CHECK_OR_RETURN(input_tensor->is_eager());  // NOLINT(maybe-need-error-msg)

  std::shared_ptr<Scope> scope = JUST(NewScopeWithParallelDescByTensor(input_tensor));

  OperatorConf op_conf;
  op_conf.set_name(op_expr.op_name());  // construct by python nn.Graph
  op_conf.set_scope_symbol_id(JUST(scope->symbol_id()));
  op_conf.set_device_tag(JUST(GetDeviceTagOfTensor(input_tensor)));
  // NOTE(chengcheng):
  //   We contruct VariableOpConf instead of FeedVariableOpConf because FeedVariableOpExpr JUST
  //   for getting input EagerTensor.
  VariableOpConf* var_conf = op_conf.mutable_variable_conf();
  var_conf->set_out("out");
  input_tensor->shape()->ToProto(var_conf->mutable_shape());
  var_conf->set_data_type(input_tensor->dtype()->data_type());
  // NOTE(chengcheng): VariableOpConf initializer_conf is useless because variable is inited
  //   by EagerTensor.
  var_conf->mutable_initializer()->mutable_empty_conf();
  JUST(GenVariableOpConfNdSbpStringByTensor(var_conf, input_tensor));
  if (!input_tensor->requires_grad()) { var_conf->set_trainable(false); }
  if (input_tensor->requires_grad()) {
    double l2 = JUST(ctx.attrs.GetAttr<double>("l2"));
    if (unlikely(l2 != 0.0)) { var_conf->mutable_regularizer()->mutable_l1_l2_conf()->set_l2(l2); }
  }

  auto infer_ctx = JUST(GetCurInferCtx());
  VLOG(2) << "Lazy nn.Graph name " << infer_ctx->job().job_conf().job_name()
          << " try to add op: \n: " << op_conf.DebugString() << std::endl;
  OpAttribute op_attr = *JUST(infer_ctx->AddAndInferGlobalOp(op_conf));
  VLOG(2) << "Lazy nn.Graph name " << infer_ctx->job().job_conf().job_name() << " add op : \n"
          << op_conf.name() << std::endl;
  VLOG(3) << "Lazy nn.Graph name " << infer_ctx->job().job_conf().job_name()
          << " infer and and op attr : \n"
          << op_attr.DebugString() << std::endl;

  int64_t parallel_desc_sym_id = JUST(scope->GetParallelDescSymbolId(op_conf));
  auto blob_parallel_desc = JUST(GetSymbol<ParallelDesc>(parallel_desc_sym_id));

  // Check outputs num and setup output tensor properties.
  CHECK_EQ_OR_RETURN(outputs->size(), 1);        // NOLINT(maybe-need-error-msg)
  CHECK_EQ_OR_RETURN(op_expr.output_size(), 1);  // NOLINT(maybe-need-error-msg)
  CHECK_OR_RETURN(!(*outputs)[0]);               // NOLINT(maybe-need-error-msg)

  const std::string obn = "out";  // NOTE(chengcheng): obn is NOT op_expr.indexed_obns
  auto origin_var = JUST(BuildTensor(op_attr, obn, blob_parallel_desc, /* is_lazy= */ true,
                                     /* is_local */ input_tensor->is_local()));
  // NOTE(chengcheng): Record variable op output LazyTenosr
  TensorNameScope::Global()->Record(origin_var, GenLogicalBlobName(op_conf.name(), obn));
  // NOTE(chengcheng): Record EagerTensor as variable tensor name
  TensorNameScope::Global()->Record(input_tensor, GenLogicalBlobName(op_conf.name(), obn));

  // NOTE: The output variable will then be repeat in DispatchFeedVariableOpExprFunctor
  // if GradAcc is enabled
  (*outputs)[0] = origin_var;
  return Maybe<void>::Ok();
}

Maybe<void> LazyInterpreter::ApplyImpl(const FetchOutputOpExpr& op_expr, const TensorTuple& inputs,
                                       TensorTuple* outputs, const OpExprInterpContext& ctx) const {
  // NOTE: The input has been packed in DispatchFetchOutputOpExprFunctor
  // if GradAcc is enabled
  // NOTE(chengcheng): inputs[0] is the LazyTensor
  CHECK_EQ_OR_RETURN(inputs.size(), 1);         // NOLINT(maybe-need-error-msg)
  CHECK_EQ_OR_RETURN(op_expr.input_size(), 1);  // NOLINT(maybe-need-error-msg)
  const std::shared_ptr<Tensor>& input_tensor = inputs.at(0);
  std::string input_lbn = TensorNameScope::Global()->Lookup(input_tensor);
  // Lazy tensor must has lbn.
  // Eager tensor may has lbn if it has already been treated as an output of a variable op
  // or an output of an inplace op.
  if (input_lbn.empty()) {
    CHECK_OR_RETURN(input_tensor->is_eager());  // NOLINT(maybe-need-error-msg)
    // This output tensor is a new free eager tensor, so treat it as a new variable op output.
    JUST(AddFreeEagerTensorToVariableOp(input_tensor));
    input_lbn = TensorNameScope::Global()->Lookup(input_tensor);
    CHECK_OR_RETURN(!input_lbn.empty());  // NOLINT(maybe-need-error-msg)
  }
  std::shared_ptr<Scope> scope = JUST(NewScopeWithParallelDescByTensor(input_tensor));

  OperatorConf op_conf;
  op_conf.set_name(op_expr.op_name());  // construct by python nn.Graph
  op_conf.set_scope_symbol_id(JUST(scope->symbol_id()));
  op_conf.set_device_tag(JUST(GetDeviceTagOfTensor(input_tensor)));
  // NOTE(chengcheng):
  //   We contruct OutputOpConf instead of FetchOutputOpConf because FetchOutputOpExpr JUST
  //   for get nn.Graph output LazyTensor.
  OutputOpConf* output_conf = op_conf.mutable_output_conf();
  output_conf->set_in(input_lbn);
  output_conf->set_out("out");
  InterfaceBlobConf* blob_conf = output_conf->mutable_blob_conf();
  input_tensor->shape()->ToProto(blob_conf->mutable_shape());
  blob_conf->set_data_type(input_tensor->dtype()->data_type());
  // NOTE(chengcheng): is_dynamic true has conflict in global lazy job even if world size 1.
  //     this flag will be removed in the future.
  // blob_conf->set_is_dynamic(GetIsDynamicOfTensor(input_tensor));
  blob_conf->set_is_dynamic(false);
  JUST(GenNdSbpByTensor(blob_conf->mutable_nd_sbp(), input_tensor));

  auto infer_ctx = JUST(GetCurInferCtx());
  VLOG(2) << "Lazy nn.Graph name " << infer_ctx->job().job_conf().job_name() << " try to add op: \n"
          << op_conf.DebugString() << std::endl;
  OpAttribute op_attr = *JUST(infer_ctx->AddAndInferGlobalOp(op_conf));
  VLOG(2) << "Lazy nn.Graph name " << infer_ctx->job().job_conf().job_name() << " add op : \n"
          << op_conf.name() << std::endl;
  VLOG(3) << "Lazy nn.Graph name " << infer_ctx->job().job_conf().job_name()
          << " infer and and op attr : \n"
          << op_attr.DebugString() << std::endl;

  int64_t parallel_desc_sym_id = JUST(scope->GetParallelDescSymbolId(op_conf));
  auto blob_parallel_desc = JUST(GetSymbol<ParallelDesc>(parallel_desc_sym_id));

  // Check outputs num and setup output tensor properties.
  CHECK_EQ_OR_RETURN(outputs->size(), 1);        // NOLINT(maybe-need-error-msg)
  CHECK_EQ_OR_RETURN(op_expr.output_size(), 1);  // NOLINT(maybe-need-error-msg)
  CHECK_OR_RETURN(!(*outputs)[0]);               // NOLINT(maybe-need-error-msg)
  const std::string obn = "out";  // NOTE(chengcheng): obn is NOT op_expr.indexed_obns
  (*outputs)[0] = JUST(BuildTensor(op_attr, obn, blob_parallel_desc, /* is_lazy= */ false,
                                   /* is_local= */ input_tensor->is_local()));
  return Maybe<void>::Ok();
}

Maybe<void> LazyInterpreter::ApplyImpl(const ImageDecoderRandomCropResizeOpExpr& op_expr,
                                       const TensorTuple& inputs, TensorTuple* outputs,
                                       const OpExprInterpContext& ctx) const {
  CHECK_EQ_OR_RETURN(inputs.size(), 1);         // NOLINT(maybe-need-error-msg)
  CHECK_EQ_OR_RETURN(op_expr.input_size(), 1);  // NOLINT(maybe-need-error-msg)
  const std::shared_ptr<Tensor>& input_tensor = inputs.at(0);
  const std::string& input_lbn = TensorNameScope::Global()->Lookup(input_tensor);
  CHECK_OR_RETURN(!input_lbn.empty());  // NOLINT(maybe-need-error-msg)

  auto op_conf = JUST(OpInterpUtil::GenBuiltinOpConf(op_expr, ctx.attrs));
  std::string device_tag;
  if (IsCpuOnly(*op_conf)) {
    device_tag = "cpu";
  } else {
    device_tag = "cuda";
  }

  ParallelConf parallel_conf = JUST(GetParallelDescOfTensor(input_tensor))->parallel_conf();
  parallel_conf.set_device_tag(device_tag);  // NOTE(chengcheng): only support gpu decode.
  const auto& scope = JUST(NewScopeWithParallelConfAndCurScope(parallel_conf));

  op_conf->set_scope_symbol_id(JUST(scope->symbol_id()));
  op_conf->set_device_tag(device_tag);

  // NOTE(chengcheng): replace right input_lbn and obn
  ReplaceInputLbnInOpCustomizedConf(op_conf.get(), /* ibn */ "in", input_lbn);
  op_conf->mutable_image_decoder_random_crop_resize_conf()->set_out("out");

  auto infer_ctx = JUST(GetCurInferCtx());
  // NOTE(chengcheng): MUST reset unique op name before InferCtx::AddOp
  const std::string new_op_name = *JUST(infer_ctx->NewUniqueOpNameByFunctionalOpConf(*op_conf));
  op_conf->set_name(new_op_name);
  VLOG(2) << "Lazy nn.Graph name " << infer_ctx->job().job_conf().job_name() << " try to add op: \n"
          << op_conf->DebugString() << std::endl;
  OpAttribute op_attr = *JUST(infer_ctx->AddAndInferGlobalOp(*op_conf));
  VLOG(2) << "Lazy nn.Graph name " << infer_ctx->job().job_conf().job_name() << " add op : \n"
          << op_conf->name() << std::endl;
  VLOG(3) << "Lazy nn.Graph name " << infer_ctx->job().job_conf().job_name()
          << " infer and and op attr : \n"
          << op_attr.DebugString() << std::endl;

  int64_t parallel_desc_sym_id = JUST(scope->GetParallelDescSymbolId(*op_conf));
  auto blob_parallel_desc = JUST(GetSymbol<ParallelDesc>(parallel_desc_sym_id));

  // Check outputs num and setup output tensor properties.
  CHECK_EQ_OR_RETURN(outputs->size(), 1);        // NOLINT(maybe-need-error-msg)
  CHECK_EQ_OR_RETURN(op_expr.output_size(), 1);  // NOLINT(maybe-need-error-msg)
  CHECK_OR_RETURN(!(*outputs)[0]);               // NOLINT(maybe-need-error-msg)
  const std::string obn = "out";  // NOTE(chengcheng): obn is NOT op_expr.indexed_obns
  (*outputs)[0] = JUST(BuildTensor(op_attr, obn, blob_parallel_desc, /* is_lazy= */ true,
                                   /* is_local= */ input_tensor->is_local()));
  TensorNameScope::Global()->Record((*outputs)[0], GenLogicalBlobName(new_op_name, obn));
  return Maybe<void>::Ok();
}

namespace {

Maybe<void> LazyInterpreterApplyImplForSourceUserOpExpr(const UserOpExpr& op_expr,
                                                        TensorTuple* outputs,
                                                        const OpExprInterpContext& ctx) {
  NonRecursiveMetaInfoConsistencyCheckScope non_scope;
  bool is_local;
  std::shared_ptr<const ParallelDesc> parallel_desc;
  if (ctx.parallel_desc.has_value()) {
    // NOTE(chengcheng): global
    CHECK_OR_RETURN(!ctx.device.has_value());  // NOLINT(maybe-need-error-msg)
    const auto& parallel_desc_sym = JUST(ctx.parallel_desc);
    parallel_desc = parallel_desc_sym.shared_from_symbol();
    JUST(MetaInfoConsistencyCheck(parallel_desc_sym, ctx.nd_sbp, 1, /* force_check */ false));
    is_local = false;
  } else {
    // NOTE(chengcheng): local
    CHECK_OR_RETURN(!ctx.nd_sbp.has_value());  // NOLINT(maybe-need-error-msg)
    if (ctx.device.has_value()) {
      const auto& device = JUST(ctx.device);
      const auto& placement = JUST(Placement4Device(device));
      parallel_desc = placement.shared_from_symbol();
    } else {
      // NOTE(chengcheng): if functor NOT set device, using cpu device default.
      const auto& device = JUST(Device::New("cpu"));
      const auto& placement = JUST(Placement4Device(device));
      parallel_desc = placement.shared_from_symbol();
    }
    is_local = true;
  }
  const auto& parallel_conf = parallel_desc->parallel_conf();
  const auto& scope = JUST(NewScopeWithParallelConfAndCurScope(parallel_conf));
  auto op_conf = JUST(OpInterpUtil::GenBuiltinOpConf(op_expr, ctx.attrs));
  op_conf->set_scope_symbol_id(JUST(scope->symbol_id()));
  op_conf->set_device_tag(parallel_conf.device_tag());

  auto infer_ctx = JUST(GetCurInferCtx());
  // NOTE(chengcheng): MUST reset unique op name before InferCtx::AddOp
  const std::string new_op_name = *JUST(infer_ctx->NewUniqueOpNameByFunctionalOpConf(*op_conf));
  const std::string graph_name = infer_ctx->job().job_conf().job_name();

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

  JUST(GradAccTryInsertRepeatTickBeforeSource(op_conf, is_local));

  VLOG(2) << "Lazy nn.Graph name " << infer_ctx->job().job_conf().job_name() << " try to add op: \n"
          << op_conf->DebugString() << std::endl;
  OpAttribute op_attr = *JUST(infer_ctx->AddAndInferGlobalOp(*op_conf));
  VLOG(2) << "Lazy nn.Graph name " << infer_ctx->job().job_conf().job_name() << " add op : \n"
          << op_conf->name() << std::endl;
  VLOG(3) << "Lazy nn.Graph name " << infer_ctx->job().job_conf().job_name()
          << " infer and and op attr : \n"
          << op_attr.DebugString() << std::endl;

  int64_t parallel_desc_sym_id = JUST(scope->GetParallelDescSymbolId(*op_conf));
  auto blob_parallel_desc = JUST(GetSymbol<ParallelDesc>(parallel_desc_sym_id));

  // Check outputs num and setup output tensor properties.
  CHECK_EQ_OR_RETURN(outputs->size(), op_expr.output_size());  // NOLINT(maybe-need-error-msg)
  for (int i = 0; i < op_expr.output_size(); ++i) {
    const std::string& obn = op_expr.indexed_obns().at(i);
    if (!(*outputs)[i]) {
      (*outputs)[i] =
          JUST(BuildTensor(op_attr, obn, blob_parallel_desc, /* is_lazy= */ true, is_local));
    } else {
      VLOG(2) << "Lazy nn.Graph name " << graph_name << " source op name " << new_op_name
              << " run with inplace.";
      const std::shared_ptr<Tensor>& inplace_out = (*outputs)[i];
      JUST(CheckTensorMatchAttr(inplace_out, op_attr, obn, blob_parallel_desc, is_local));
    }
    TensorNameScope::Global()->Record((*outputs)[i], GenLogicalBlobName(new_op_name, obn));
  }
  return Maybe<void>::Ok();
}

Maybe<void> LazyInterpreterApplyImplForCopyUserOpExpr(const UserOpExpr& op_expr,
                                                      const TensorTuple& inputs,
                                                      TensorTuple* outputs,
                                                      const OpExprInterpContext& ctx) {
  CHECK_OR_RETURN(op_expr.op_type_name() == "copy");  // NOLINT(maybe-need-error-msg)
  CHECK_EQ_OR_RETURN(inputs.size(), 1);               // NOLINT(maybe-need-error-msg)
  CHECK_EQ_OR_RETURN(op_expr.input_size(), 1);        // NOLINT(maybe-need-error-msg)
  const std::shared_ptr<Tensor>& input_tensor = inputs.at(0);
  std::string input_lbn = TensorNameScope::Global()->Lookup(input_tensor);
  if (input_lbn.empty()) {
    JUST(AddFreeEagerTensorToVariableOp(input_tensor));
    input_lbn = TensorNameScope::Global()->Lookup(input_tensor);
  }
  CHECK_OR_RETURN(!input_lbn.empty());  // NOLINT(maybe-need-error-msg)
  auto device = JUST(ctx.attrs.GetAttr<Symbol<Device>>("device"));

  CHECK_EQ_OR_RETURN(outputs->size(), 1);        // NOLINT(maybe-need-error-msg)
  CHECK_EQ_OR_RETURN(op_expr.output_size(), 1);  // NOLINT(maybe-need-error-msg)
  if (input_tensor->is_local()) {
    (*outputs)[0] =
        JUST(LocalTensor::MakeTensor(input_tensor->shape(), JUST(input_tensor->stride()),
                                     input_tensor->dtype()->data_type(), device,
                                     /* is_lazy= */ true,
                                     /*requires_grad=*/false, /*is_leaf=*/true));
  } else {
    ParallelConf parallel_conf = JUST(input_tensor->parallel_desc())->parallel_conf();
    parallel_conf.set_device_tag(device->type());
    ParallelDesc parallel_desc(parallel_conf);
    (*outputs)[0] =
        JUST(GlobalTensor::MakeTensor(input_tensor->shape(), input_tensor->dtype()->data_type(),
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
  CHECK_EQ_OR_RETURN(inputs.size(), op_expr.input_size());  // NOLINT(maybe-need-error-msg)

  // NOTE(chengcheng): Handle special UserOp such as:
  //     1. [Source UserOp] : OFRecordReader, CoinFlip
  //     2. [Change Placement/ParallelDesc UserOp] : to(copy)/to_global/parallel_cast
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
  CHECK_GE_OR_RETURN(inputs.size(), 1);  // NOLINT(maybe-need-error-msg)
  auto op_conf = JUST(OpInterpUtil::GenBuiltinOpConf(op_expr, ctx.attrs));
  std::shared_ptr<Scope> scope = JUST(NewScopeWithParallelDescByTensor(JUST(VectorAt(inputs, 0))));
  op_conf->set_scope_symbol_id(JUST(scope->symbol_id()));
  const std::string device_tag = JUST(GetDeviceTagOfTensor(JUST(VectorAt(inputs, 0))));
  const bool is_local = inputs.at(0)->is_local();
  const std::shared_ptr<const ParallelDesc> parallel_desc =
      JUST(GetParallelDescOfTensor(inputs.at(0)));

  op_conf->set_device_tag(device_tag);
  auto infer_ctx = JUST(GetCurInferCtx());
  // NOTE(chengcheng): MUST reset unique op name before InferCtx::AddOp
  const std::string new_op_name = *JUST(infer_ctx->NewUniqueOpNameByFunctionalOpConf(*op_conf));
  const std::string graph_name = infer_ctx->job().job_conf().job_name();

  for (int i = 0; i < inputs.size(); ++i) {
    const auto& input_tensor = inputs.at(i);
    CHECK_EQ_OR_RETURN(is_local, input_tensor->is_local());  // NOLINT(maybe-need-error-msg)
    if (!op_expr.IsHostMemoryInput(i)) {
      if (is_local) {
        CHECK_OR_RETURN(device_tag == JUST(GetDeviceTagOfTensor(input_tensor)))
            << Error::RuntimeError() << "Lazy nn.Graph name: " << graph_name
            << " encountered ERROR in module/op_name: " << new_op_name
            << ". Expected all tensors to be on the same device, but found at least two devices, "
            << JUST(JUST(VectorAt(inputs, 0))->device())->ToString() << " (positional 0) and "
            << JUST(JUST(VectorAt(inputs, i))->device())->ToString() << " (positional " << i
            << ")! Please use tensor.to() to synchronize all the input with the same device.";
      } else {
        // TODO: Print out all the placement
        CHECK_OR_RETURN(parallel_desc->Equals(*JUST(GetParallelDescOfTensor(input_tensor))))
            << Error::RuntimeError() << "Lazy nn.Graph name: " << graph_name
            << " encountered ERROR in module/op_name: " << new_op_name
            << ". Expected all tensors to be on the same placement, but found at least two "
               "placements, "
            << *JUST(PlacementToString(JUST(JUST(VectorAt(inputs, 0))->parallel_desc())))
            << " (positional 0) and "
            << *JUST(PlacementToString(JUST(JUST(VectorAt(inputs, i))->parallel_desc())))
            << " (positional " << i
            << ")! Please use tensor.to_global() to synchronize all the input with the same "
               "placement.";
      }
    }
    const std::string& ibn = op_expr.indexed_ibns().at(i);
    std::string lbn = TensorNameScope::Global()->Lookup(input_tensor);
    if (lbn.empty()) {
      JUST(AddFreeEagerTensorToVariableOp(input_tensor));
      lbn = TensorNameScope::Global()->Lookup(input_tensor);
    }
    CHECK_OR_RETURN(!lbn.empty());  // NOLINT(maybe-need-error-msg)
    ReplaceInputLbnInOpCustomizedConf(op_conf.get(), ibn, lbn);
  }

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

  // Check outputs num and setup output tensor properties.
  CHECK_EQ_OR_RETURN(outputs->size(), op_expr.output_size());  // NOLINT(maybe-need-error-msg)

  // Disable boxing if the computation is inplace.
  for (int i = 0; i < op_expr.output_size(); ++i) {
    const auto& output = outputs->at(i);
    if (output) {
      const std::string& lbn = TensorNameScope::Global()->Lookup(output);
      CHECK_OR_RETURN(!lbn.empty()) << "The output which index is " << i
                                    << " has no tensor name, please check whether the inplaced "
                                       "output is also an input of the operation "
                                    << new_op_name;
      JUST(infer_ctx->DisableBoxing(lbn));
    }
  }
  VLOG(2) << "Lazy nn.Graph name " << graph_name << " try to add op: \n"
          << op_conf->DebugString() << std::endl;
  OpAttribute op_attr = *JUST(infer_ctx->AddAndInferGlobalOp(*op_conf));
  VLOG(2) << "Lazy nn.Graph name " << graph_name << " add op : \n" << op_conf->name() << std::endl;
  VLOG(3) << "Lazy nn.Graph name " << graph_name << " infer and and op attr : \n"
          << op_attr.DebugString() << std::endl;

  int64_t parallel_desc_sym_id = JUST(scope->GetParallelDescSymbolId(*op_conf));
  auto blob_parallel_desc = JUST(GetSymbol<ParallelDesc>(parallel_desc_sym_id));
  for (int i = 0; i < op_expr.output_size(); ++i) {
    const std::string& obn = op_expr.indexed_obns().at(i);
    if (!(*outputs)[i]) {
      (*outputs)[i] =
          JUST(BuildTensor(op_attr, obn, blob_parallel_desc, /* is_lazy= */ true, is_local));
    } else {
      VLOG(2) << "Lazy nn.Graph name " << graph_name << " op name " << new_op_name
              << " run with inplace.";
      const std::shared_ptr<Tensor>& inplace_out = (*outputs)[i];
      JUST(CheckTensorMatchAttr(inplace_out, op_attr, obn, blob_parallel_desc, is_local));
    }
    TensorNameScope::Global()->Record((*outputs)[i], GenLogicalBlobName(new_op_name, obn));
  }
  return Maybe<void>::Ok();
}

Maybe<void> LazyInterpreter::ApplyImpl(const FunctionOpExpr& op_expr, const TensorTuple& inputs,
                                       TensorTuple* outputs, const OpExprInterpContext&) const {
  // Must reset ctx in each forward
  op_expr.reset_state();
  std::shared_ptr<FunctionAutoGradCaptureState> ctx = op_expr.state();
  *outputs = *(op_expr.forward()(ctx, inputs));
  return Maybe<void>::Ok();
}

Maybe<void> LazyInterpreter::ApplyImpl(const GlobalToGlobalOpExpr& op_expr,
                                       const TensorTuple& inputs, TensorTuple* outputs,
                                       const OpExprInterpContext& ctx) const {
  CHECK_EQ_OR_RETURN(op_expr.input_size(), 1);  // NOLINT(maybe-need-error-msg)
  CHECK_EQ_OR_RETURN(inputs.size(), 1);         // NOLINT(maybe-need-error-msg)
  const auto& input_tensor = inputs[0];
  CHECK_OR_RETURN(input_tensor->is_global());  // NOLINT(maybe-need-error-msg)

  CHECK_OR_RETURN(ctx.parallel_desc.has_value());  // NOLINT(maybe-need-error-msg)
  const auto& parallel_desc_sym = JUST(ctx.parallel_desc);
  CHECK_OR_RETURN(ctx.nd_sbp.has_value());  // NOLINT(maybe-need-error-msg)
  const auto& sbp_sym = JUST(ctx.nd_sbp);

  std::string input_lbn = TensorNameScope::Global()->Lookup(input_tensor);
  if (input_lbn.empty()) {
    JUST(AddFreeEagerTensorToVariableOp(input_tensor));
    input_lbn = TensorNameScope::Global()->Lookup(input_tensor);
    CHECK_OR_RETURN(!input_lbn.empty());  // NOLINT(maybe-need-error-msg)
  }

  std::shared_ptr<Tensor> input_proxy;
  if (!JUST(GetParallelDescOfTensor(input_tensor))
           ->Equals(*parallel_desc_sym.shared_from_symbol())) {
    // NOTE(zwx): The input tensor's parallel_desc is not equal to that of op's,
    // create a proxy input with the parallel_desc that is the same as op's
    input_proxy =
        JUST(GlobalTensor::MakeTensor(input_tensor->shape(), input_tensor->dtype()->data_type(),
                                      JUST(input_tensor->nd_sbp()), parallel_desc_sym,
                                      /* is_lazy= */ true,
                                      /*requires_grad=*/false, /*is_leaf=*/true));
    TensorNameScope::Global()->Record(input_proxy, input_lbn);
  }

  CHECK_EQ_OR_RETURN(op_expr.output_size(), 1);  // NOLINT(maybe-need-error-msg)
  CHECK_EQ_OR_RETURN(outputs->size(), 1);        // NOLINT(maybe-need-error-msg)
  CHECK_OR_RETURN(!(*outputs)[0]);               // NOLINT(maybe-need-error-msg)

  if (!op_expr.grad_nd_sbp().has_value() && sbp_sym == JUST(input_tensor->nd_sbp())) {
    // NOTE(chengcheng):  if to_global ONLY change placement (nd_sbp and grad_nd_sbp is same),
    //    there is no need to build hierarchical_parallel_cast op.
    if (input_proxy) {
      (*outputs)[0] = input_proxy;
    } else {
      (*outputs)[0] = input_tensor;
    }
    return Maybe<void>::Ok();
  }

  // build parallel cast op expr
  std::shared_ptr<std::vector<std::string>> sbp_list_ptr = JUST(GetNdSbpStrList(sbp_sym));
  std::string grad_mode;
  std::vector<std::string> grad_sbp_str_list;
  if (op_expr.grad_nd_sbp().has_value()) {
    grad_mode = "manual";
    grad_sbp_str_list = *JUST(GetNdSbpStrList(JUST(op_expr.grad_nd_sbp())));
  } else {
    grad_mode = "identity";
  }
  std::shared_ptr<UserOpExpr> parallel_cast_op_expr =
      JUST(OpBuilder("hierarchical_parallel_cast", "trivial_op_name")
               .Input("in")
               .Output("out")
               .Attr<std::vector<std::string>>("nd_sbp", *sbp_list_ptr)
               .Attr<std::string>("grad_mode", grad_mode)
               .Attr<std::vector<std::string>>("grad_nd_sbp", grad_sbp_str_list)
               .Build());

  if (input_proxy) {
    (*outputs)[0] =
        JUST(OpInterpUtil::Dispatch<one::Tensor>(*parallel_cast_op_expr, {input_proxy}));
  } else {
    (*outputs)[0] =
        JUST(OpInterpUtil::Dispatch<one::Tensor>(*parallel_cast_op_expr, {input_tensor}));
  }

  return Maybe<void>::Ok();
}

}  // namespace one
}  // namespace oneflow
