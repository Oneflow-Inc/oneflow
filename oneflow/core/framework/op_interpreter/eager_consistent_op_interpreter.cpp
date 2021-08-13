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

#include <sstream>
#include "oneflow/core/framework/id_util.h"
#include "oneflow/core/framework/to_string.h"
#include "oneflow/core/framework/op_interpreter.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/framework/op_arg_util.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/framework/scope_util.h"
#include "oneflow/core/framework/session_util.h"
#include "oneflow/core/framework/symbol_storage_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_name_scope.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/framework/consistent_tensor_infer_cache.h"
#include "oneflow/core/eager/foreign_boxing_util.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/autograd/autograd_mode.h"
#include "oneflow/core/framework/op_interpreter/boxing/eager_boxing_interpreter_mgr.h"
#include "oneflow/user/kernels/stateful_local_opkernel.h"
#include "oneflow/core/framework/tensor_rpc_util.h"
#include "oneflow/core/framework/tensor_consistent_id.h"
#include "oneflow/core/common/decorator.h"

namespace oneflow {
namespace one {

namespace {

Maybe<Symbol<cfg::ParallelDistribution>> MakeBroadcastParallelDesc() {
  cfg::ParallelDistribution broadcast_parallel_desc;
  broadcast_parallel_desc.mutable_sbp_parallel()->Add()->mutable_broadcast_parallel();
  return SymbolOf(broadcast_parallel_desc);
}

Maybe<UserOpExpr> FindOrCreatHierarchicalParallelCastOpExpr(
    Symbol<cfg::ParallelDistribution> nd_sbp) {
  thread_local HashMap<Symbol<cfg::ParallelDistribution>, std::shared_ptr<UserOpExpr>>
      nd_sbp2hierarchical_parallel_cast_op_expr;
  auto iter = nd_sbp2hierarchical_parallel_cast_op_expr.find(nd_sbp);
  if (iter == nd_sbp2hierarchical_parallel_cast_op_expr.end()) {
    std::shared_ptr<UserOpExpr> op_expr =
        JUST(OpBuilder("hierarchical_parallel_cast",
                       *CHECK_JUST(UniqueStr("hierarchical_parallel_cast")))
                 .Input("in")
                 .Output("out")
                 .Attr<std::vector<std::string>>("nd_sbp", *JUST(GetNdSbpStrList(nd_sbp)))
                 .Attr<std::string>("grad_mode", "restore")
                 .Attr<std::vector<std::string>>("grad_nd_sbp", std::vector<std::string>())
                 .Build());
    iter = nd_sbp2hierarchical_parallel_cast_op_expr.emplace(nd_sbp, op_expr).first;
  }
  return iter->second;
}

Maybe<UserOpExpr> EagerNcclBroadcast(Symbol<ParallelDesc> parallel_desc, int64_t root) {
  return OpBuilder("eager_nccl_broadcast", *JUST(UniqueStr("eager_nccl_broadcast")))
      .Input("in")
      .Output("out")
      .Attr<std::string>("parallel_conf", PbMessage2TxtString(parallel_desc->parallel_conf()))
      .Attr<int64_t>("root", root)
      .Build();
}

Maybe<UserOpExpr> FindOrCreatEagerNcclBroadcastOpExpr(Symbol<ParallelDesc> parallel_desc) {
  static thread_local HashMap<Symbol<ParallelDesc>, std::shared_ptr<UserOpExpr>>
      parallel_desc2eager_nccl_broadcast;
  auto iter = parallel_desc2eager_nccl_broadcast.find(parallel_desc);
  if (iter == parallel_desc2eager_nccl_broadcast.end()) {
    int64_t root = JUST(parallel_desc->MachineId4ParallelId(0));
    std::shared_ptr<UserOpExpr> op_expr = JUST(EagerNcclBroadcast(parallel_desc, root));
    iter = parallel_desc2eager_nccl_broadcast.emplace(parallel_desc, op_expr).first;
  }
  return iter->second;
}

Maybe<Symbol<ParallelDesc>> GetParallelDesc(const TensorTuple& inputs,
                                            const OpExprInterpContext& ctx) {
  if (!inputs.empty()) { return inputs.at(0)->parallel_desc(); }
  return ctx.parallel_desc.value();
}

std::string GetDynamicOpConsistentFailedDebugString(const UserOpExpr& user_op_expr,
                                                    const StatefulLocalOpKernel& kernel) {
  CHECK(!kernel.output_tuple_indexes4mut2_obns().empty());
  std::string plentysuffix = kernel.output_tuple_indexes4mut2_obns().size() == 1 ? "s" : "";
  std::stringstream ss;
  ss << "operator `" << user_op_expr.op_type_name() << "`"
     << " does not support consistent mode because the shape" << plentysuffix << " of output tensor"
     << plentysuffix << " ";
  int i = 0;
  for (const auto& out_index : kernel.output_tuple_indexes4mut2_obns()) {
    if (i++ > 0) { ss << ", "; }
    ss << out_index;
  }
  ss << " are not infered before op computation.";
  return ss.str();
}

Maybe<Tensor> CalcBoxingOutput(const std::shared_ptr<Tensor>& input,
                               Symbol<cfg::ParallelDistribution> out_nd_sbp,
                               bool current_rank_local_is_valid) {
  if (!current_rank_local_is_valid) { return input; }
  const auto* mgr = Global<EagerBoxingInterpreterManager>::Get();
  // Eager boxing
  const auto& in_nd_sbp = JUST(input->nd_sbp());
  const auto& in_parallel_desc = JUST(input->parallel_desc());
  const auto& out_parallel_desc = in_parallel_desc;
  const auto& boxing_interpreter = JUST(
      mgr->GetEagerBoxingInterpreter(in_nd_sbp, out_nd_sbp, in_parallel_desc, out_parallel_desc));
  const auto& output = JUST(boxing_interpreter->Interpret(input, in_nd_sbp, out_nd_sbp,
                                                          in_parallel_desc, out_parallel_desc));
  return output;
}

auto* GetBoxingOutput =
    DECORATE(DECORATE(&CalcBoxingOutput, CheckConsistentTensorMeta), DisableRecusiveBoxingCall);

Maybe<void> RunInstructionLocalCallOpKernel(
    const UserOpExpr& user_op_expr, const TensorTuple& inputs, TensorTuple* outputs,
    const OpExprInterpContext& ctx,
    const std::shared_ptr<const one::ConsistentTensorInferResult>& result, Symbol<Device> device,
    Symbol<ParallelDesc> parallel_desc, const Optional<int64_t>& parallel_id, bool allow_boxing) {
  const auto& kernel = JUST(user_op_expr.MutKernel4Device(*device));
  CHECK_EQ_OR_RETURN(kernel->output_tuple_indexes4mut2_obns().size(), 0)
      << Error::Unimplemented() << GetDynamicOpConsistentFailedDebugString(user_op_expr, *kernel);
  std::shared_ptr<EagerBlobObjectList> input_eager_blob_objects =
      std::make_shared<EagerBlobObjectList>(inputs.size());
  for (int i = 0; i < inputs.size(); ++i) {
    std::shared_ptr<Tensor> input = inputs.at(i);
    const auto& infered_input_meta = result->input_tensor_metas().at(i);
    if (allow_boxing && infered_input_meta->nd_sbp() != JUST(input->nd_sbp())) {
      input = JUST(GetBoxingOutput(input, infered_input_meta->nd_sbp(), parallel_id.has_value()));
    }
    const auto& local_tensor = JUST(input->cur_rank_phy_tensor());
    input_eager_blob_objects->at(i) = JUST(local_tensor->eager_blob_object());
  }
  // Do nothing if the `parallel_desc` doesn't cover current ProcessCtx.
  if (!parallel_id.has_value()) { return Maybe<void>::Ok(); }
  std::shared_ptr<EagerBlobObjectList> output_eager_blob_objects =
      std::make_shared<EagerBlobObjectList>(outputs->size());
  for (int i = 0; i < outputs->size(); ++i) {
    const auto& local_tensor = JUST(outputs->at(i)->cur_rank_phy_tensor());
    output_eager_blob_objects->at(i) = JUST(local_tensor->eager_blob_object());
  }
  const auto& instr_type_name = JUST(GetLocalCallInstructionName(parallel_desc->device_tag()));
  JUST(PhysicalRun([&](InstructionsBuilder* builder) -> Maybe<void> {
    return builder->LocalCallOpKernel(kernel, input_eager_blob_objects, output_eager_blob_objects,
                                      result, ctx, parallel_desc.shared_from_symbol(),
                                      instr_type_name);
  }));
  return Maybe<void>::Ok();
}

Maybe<void> Interpret(const UserOpExpr& user_op_expr, const TensorTuple& inputs,
                      TensorTuple* outputs, const OpExprInterpContext& ctx) {
  CHECK_EQ_OR_RETURN(outputs->size(), user_op_expr.output_size());
  const auto& parallel_desc = JUST(GetParallelDesc(inputs, ctx));
  std::shared_ptr<const ConsistentTensorInferResult> result;
  if (inputs.empty()) {
    const auto& infer_args = JUST(SrcOpConsistentTensorMetaInferArgs::New(
        ctx.attrs, parallel_desc, JUST(ctx.nd_sbp.value())));
    result = JUST(user_op_expr.mut_consistent_tensor_infer_cache()->GetOrInfer(*infer_args));
  } else {
    const auto& infer_args =
        JUST(ConsistentTensorMetaInferArgs::New(ctx.attrs, inputs, parallel_desc));
    result = JUST(user_op_expr.mut_consistent_tensor_infer_cache()->GetOrInfer(*infer_args));
  }
  const auto& output_tensor_metas = result->output_tensor_metas();
  Optional<int64_t> parallel_id;
  const auto& device = JUST(GetDevice4CurrentProcessCtx(parallel_desc, &parallel_id));
  for (int i = 0; i < outputs->size(); ++i) {
    const auto& tensor_impl = JUST(EagerConsistentTensorImpl::New(output_tensor_metas.at(i), device,
                                                                  parallel_id, false, false));
    outputs->at(i).reset(new ConsistentTensor(tensor_impl));
  }
  // Run instruction LocalCallOpKernel
  JUST(RunInstructionLocalCallOpKernel(user_op_expr, inputs, outputs, ctx, result, device,
                                       parallel_desc, parallel_id, /* allow_boxing */ true));
  return Maybe<void>::Ok();
}

auto* InterpretThenInitConsistentId = DECORATE(&Interpret, NonRecursiveInitConsistentId);

}  // namespace

Maybe<void> EagerConsistentInterpreter::ApplyImpl(const UserOpExpr& op_expr,
                                                  const TensorTuple& inputs, TensorTuple* outputs,
                                                  const OpExprInterpContext& ctx) const {
  return InterpretThenInitConsistentId(op_expr, inputs, outputs, ctx);
}

Maybe<void> EagerConsistentInterpreter::ApplyImpl(const VariableOpExpr& op_expr,
                                                  const TensorTuple& inputs, TensorTuple* outputs,
                                                  const OpExprInterpContext& ctx) const {
  OF_UNIMPLEMENTED();
}

Maybe<void> EagerConsistentInterpreter::ApplyImpl(const ConsistentToConsistentOpExpr& op_expr,
                                                  const TensorTuple& inputs, TensorTuple* outputs,
                                                  const OpExprInterpContext& ctx) const {
  CHECK_EQ_OR_RETURN(inputs.size(), 1);
  CHECK_OR_RETURN(ctx.parallel_desc.has_value());
  CHECK_OR_RETURN(ctx.nd_sbp.has_value());
  const auto& output_parallel_desc = JUST(ctx.parallel_desc.value());
  const auto& input_parallel_desc = JUST(inputs.at(0)->parallel_desc());
  const auto& nd_sbp_cast_op_expr =
      JUST(FindOrCreatHierarchicalParallelCastOpExpr(JUST(ctx.nd_sbp.value())));
  if (output_parallel_desc == input_parallel_desc) {
    outputs->at(0) = JUST(OpInterpUtil::Dispatch<Tensor>(*nd_sbp_cast_op_expr, inputs));
    return Maybe<void>::Ok();
  } else {
    static Symbol<cfg::ParallelDistribution> broadcast_nd_sbp = JUST(MakeBroadcastParallelDesc());
    const auto& broadcast_nd_sbp_cast_op_expr =
        JUST(FindOrCreatHierarchicalParallelCastOpExpr(broadcast_nd_sbp));
    std::shared_ptr<TensorTuple> broadcasted_inputs =
        JUST(OpInterpUtil::Dispatch<TensorTuple>(*broadcast_nd_sbp_cast_op_expr, inputs));
    const auto& broadcast_op_expr = JUST(FindOrCreatEagerNcclBroadcastOpExpr(output_parallel_desc));

    const auto& infer_args = JUST(
        ConsistentTensorMetaInferArgs::New(AttrMap{}, *broadcasted_inputs, output_parallel_desc));
    std::shared_ptr<const ConsistentTensorInferResult> result =
        JUST(broadcast_op_expr->mut_consistent_tensor_infer_cache()->GetOrInfer(*infer_args));
    TensorTuple broadcasted_outputs(outputs->size());
    const auto& output_tensor_metas = result->output_tensor_metas();
    Optional<int64_t> parallel_id;
    const auto& device = JUST(GetDevice4CurrentProcessCtx(output_parallel_desc, &parallel_id));
    for (int i = 0; i < broadcasted_outputs.size(); ++i) {
      const auto& tensor_impl = JUST(EagerConsistentTensorImpl::New(
          output_tensor_metas.at(i), device, parallel_id, false, false));
      const auto& transport_token = JUST(TransportToken::NewMetaTransportToken());
      JUST(tensor_impl->set_transport_token(transport_token));
      broadcasted_outputs.at(i).reset(new ConsistentTensor(tensor_impl));
    }
    // Run instruction LocalCallOpKernel
    JUST(RunInstructionLocalCallOpKernel(
        *broadcast_op_expr, *broadcasted_inputs, &broadcasted_outputs, ctx, result, device,
        output_parallel_desc, parallel_id, /* allow_boxing */ false));
    outputs->at(0) =
        JUST(OpInterpUtil::Dispatch<Tensor>(*nd_sbp_cast_op_expr, broadcasted_outputs));
    return Maybe<void>::Ok();
  }
}

Maybe<void> EagerConsistentInterpreter::ApplyImpl(const CastToConsistentOpExpr& op_expr,
                                                  const TensorTuple& inputs, TensorTuple* outputs,
                                                  const OpExprInterpContext& ctx) const {
  OF_UNIMPLEMENTED();
}

Maybe<void> EagerConsistentInterpreter::ApplyImpl(const CastFromConsistentOpExpr& op_expr,
                                                  const TensorTuple& inputs, TensorTuple* outputs,
                                                  const OpExprInterpContext& ctx) const {
  CHECK_EQ_OR_RETURN(inputs.size(), 1);
  const auto& input_tensor = inputs.at(0);
  const auto& mirrored_tensor = JUST(JUST(input_tensor->cur_rank_phy_tensor())->detach());
  bool requires_grad = autograd::GradMode::is_enabled() && input_tensor->requires_grad();
  mirrored_tensor->set_requires_grad(requires_grad);
  mirrored_tensor->set_is_leaf(!requires_grad);
  outputs->at(0) = mirrored_tensor;
  return Maybe<void>::Ok();
}

Maybe<void> EagerConsistentInterpreter::ApplyImpl(const CastToMirroredOpExpr& op_expr,
                                                  const TensorTuple& inputs, TensorTuple* outputs,
                                                  const OpExprInterpContext& ctx) const {
  OF_UNIMPLEMENTED();
}

Maybe<void> EagerConsistentInterpreter::ApplyImpl(const CastFromMirroredOpExpr& op_expr,
                                                  const TensorTuple& inputs, TensorTuple* outputs,
                                                  const OpExprInterpContext& ctx) const {
  OF_UNIMPLEMENTED();
}

Maybe<void> EagerConsistentInterpreter::ApplyImpl(const DistributeSplitOpExpr& op_expr,
                                                  const TensorTuple& inputs, TensorTuple* outputs,
                                                  const OpExprInterpContext& ctx) const {
  OF_UNIMPLEMENTED();
}

Maybe<void> EagerConsistentInterpreter::ApplyImpl(const DistributeCloneOpExpr& op_expr,
                                                  const TensorTuple& inputs, TensorTuple* outputs,
                                                  const OpExprInterpContext& ctx) const {
  OF_UNIMPLEMENTED();
}

Maybe<void> EagerConsistentInterpreter::ApplyImpl(const DistributeConcatOpExpr& op_expr,
                                                  const TensorTuple& inputs, TensorTuple* outputs,
                                                  const OpExprInterpContext& ctx) const {
  OF_UNIMPLEMENTED();
}

Maybe<void> EagerConsistentInterpreter::ApplyImpl(const DistributeAddOpExpr& op_expr,
                                                  const TensorTuple& inputs, TensorTuple* outputs,
                                                  const OpExprInterpContext& ctx) const {
  OF_UNIMPLEMENTED();
}

Maybe<void> EagerConsistentInterpreter::ApplyImpl(const SelectFirstOpExpr& op_expr,
                                                  const TensorTuple& inputs, TensorTuple* outputs,
                                                  const OpExprInterpContext& ctx) const {
  OF_UNIMPLEMENTED();
}

}  // namespace one
}  // namespace oneflow
