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
#include "oneflow/core/framework/to_string.h"
#include "oneflow/core/framework/op_interpreter.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/framework/scope_util.h"
#include "oneflow/core/framework/session_util.h"
#include "oneflow/core/framework/symbol_storage_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_name_scope.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/framework/global_tensor_infer_cache.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/autograd/autograd_mode.h"
#include "oneflow/core/boxing/eager_boxing_interpreter_mgr.h"
#include "oneflow/user/kernels/stateful_opkernel.h"
#include "oneflow/core/framework/consistency_check.h"
#include "oneflow/core/framework/tensor_rpc_util.h"
#include "oneflow/core/framework/tensor_global_id.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/common/decorator.h"
#include "oneflow/core/boxing/eager_boxing_logger.h"
#include "oneflow/core/common/cpp_attribute.h"

namespace oneflow {
namespace one {

namespace {

// NOTE: use env variable 'ONEFLOW_DISABLE_VIEW' control use view mechanism or not
// If  set true, then do not use view mechanism(and view ops)
bool IsEnvEnablePipelineParallelismAutoToGlobal() {
  static const bool env_enable_auto_to_global =
      ParseBooleanFromEnv("ONEFLOW_ENABLE_PIPELINE_PARALLELISM_AUTO_TO_GLOBAL", false);
  return env_enable_auto_to_global;
}

Maybe<bool> IsInputParallelDescIdentical(const GlobalTensorMetaInferArgs& infer_args,
                                         const UserOpExpr& user_op_expr) {
  if (infer_args.input_global_tensor_metas().empty()) { return true; }
  Symbol<ParallelDesc> default_parallel_desc;
  for (int i = 0; i < infer_args.input_global_tensor_metas().size(); ++i) {
    if (user_op_expr.IsHostMemoryInput(i)) { continue; }
    default_parallel_desc =
        JUST(VectorAt(infer_args.input_global_tensor_metas(), i)).tensor_meta()->parallel_desc();
    break;
  }
  printf("\ndefault_parallel_desc  >>>>>>>>>>> %s\n",
         JUST(PlacementToString(default_parallel_desc))->c_str());

  for (int i = 0; i < infer_args.input_global_tensor_metas().size(); ++i) {
    if (user_op_expr.IsHostMemoryInput(i)) { continue; }
    if (default_parallel_desc
        != JUST(VectorAt(infer_args.input_global_tensor_metas(), i))
               .tensor_meta()
               ->parallel_desc()) {
      printf("\ninputs i:%d parallel_desc >>>>>>>>>>> %s\n", i,
             JUST(PlacementToString(JUST(VectorAt(infer_args.input_global_tensor_metas(), i))
                                        .tensor_meta()
                                        ->parallel_desc()))
                 ->c_str());
      return false;
    }
  }
  return true;
}

Maybe<int> GetMaxRankId(Symbol<ParallelDesc> placement) {
  // const std::string& device_type = placement->device_tag();
  std::vector<int64_t> sorted_node_ids;
  sorted_node_ids.reserve(placement->sorted_machine_ids().size());
  HashMap<int64_t, std::vector<int64_t>> node_id2sorted_dev_phy_ids;
  for (int64_t machine_id : placement->sorted_machine_ids()) {
    int64_t node_id = GlobalProcessCtx::NodeId(machine_id);
    if (!std::count(sorted_node_ids.begin(), sorted_node_ids.end(), node_id)) {
      sorted_node_ids.emplace_back(node_id);
    }
    for (int64_t device_id : placement->sorted_dev_phy_ids(machine_id)) {
      node_id2sorted_dev_phy_ids[node_id].emplace_back(device_id);
    }
  }
  int64_t node_id = sorted_node_ids.at(sorted_node_ids.size() - 1);
  return node_id * GlobalProcessCtx::NumOfProcessPerNode()
         + node_id2sorted_dev_phy_ids.at(node_id).at(node_id2sorted_dev_phy_ids.at(node_id).size()
                                                     - 1);
}

Maybe<int> GetMaxRankTensorId(const TensorTuple& inputs) {
  int64_t max_rank_tensor_id = 0;
  if (inputs.size() < 0) {
    return -1;
  } else if (inputs.size() == 0) {
    return max_rank_tensor_id;
  }
  int64_t max_rank = 0;
  for (int64_t i = 0; i < inputs.size(); ++i) {
    int64_t tensor_max_rank = JUST(GetMaxRankId(JUST(inputs[i]->parallel_desc())));
    if (tensor_max_rank >= max_rank) {
      max_rank = tensor_max_rank;
      max_rank_tensor_id = i;
    }
  }
  return max_rank_tensor_id;
}

Maybe<Symbol<ParallelDesc>> GetParallelDesc(const TensorTuple& inputs,
                                            const OpExprInterpContext& ctx,
                                            const UserOpExpr& user_op_expr) {
  if (!inputs.empty()) {
    for (int32_t i = 0; i < inputs.size(); ++i) {
      if (!user_op_expr.IsHostMemoryInput(i)) { return inputs.at(i)->parallel_desc(); }
    }
  }
  return JUST(ctx.parallel_desc);
}

std::string GetDynamicOpGlobalFailedDebugString(const UserOpExpr& user_op_expr,
                                                const StatefulOpKernel& kernel) {
  CHECK(!kernel.output_tuple_indexes4mut2_obns().empty());
  std::string plentysuffix = kernel.output_tuple_indexes4mut2_obns().size() == 1 ? "s" : "";
  std::stringstream ss;
  ss << "operator `" << user_op_expr.op_type_name() << "`"
     << " does not support global mode because the shape" << plentysuffix << " of output tensor"
     << plentysuffix << " ";
  int i = 0;
  for (const auto& out_index : kernel.output_tuple_indexes4mut2_obns()) {
    if (i++ > 0) { ss << ", "; }
    ss << out_index;
  }
  ss << " are not infered before op computation.";
  return ss.str();
}

Maybe<bool> IsAllZeroSizeTensorMeta(const std::vector<Symbol<GlobalTensorMeta>>& tensor_metas) {
  if (tensor_metas.empty()) { return false; }
  for (const auto& tensor_meta : tensor_metas) {
    if (tensor_meta->shape().elem_cnt() != 0) { return false; }
  }
  return true;
}

constexpr auto* CachedIsAllZeroSizeTensorMeta =
    DECORATE(&IsAllZeroSizeTensorMeta, ThreadLocalCopiable);

Maybe<Tensor> CalcBoxingOutput(const std::shared_ptr<Tensor>& input, Symbol<NdSbp> out_nd_sbp,
                               Symbol<ParallelDesc> out_parallel_desc,
                               bool current_rank_local_is_valid) {
  const auto& logical_shape = input->shape();
  // If the input is a tensor of size 0, construct the output directly.
  if (unlikely(logical_shape->elem_cnt() == 0)) {
    GlobalTensorMeta tensor_meta(*logical_shape, input->dtype()->data_type(), out_nd_sbp,
                                 out_parallel_desc);
    const auto& tensor_impl =
        JUST(EagerGlobalTensorImpl::New(SymbolOf(tensor_meta), input->requires_grad(), false));
    std::shared_ptr<Tensor> output = std::make_shared<GlobalTensor>(tensor_impl);
    return output;
  }
  const auto* mgr = Singleton<EagerBoxingInterpreterManager>::Get();
  // Eager boxing
  const auto& in_nd_sbp = JUST(input->nd_sbp());
  const auto& in_parallel_desc = JUST(input->parallel_desc());
  const auto& boxing_interpreter = JUST(mgr->GetEagerBoxingInterpreter(
      in_nd_sbp, out_nd_sbp, in_parallel_desc, out_parallel_desc, *logical_shape));
  Singleton<const EagerBoxingLogger>::Get()->Log(
      *JUST(boxing_interpreter->boxing_interpreter_status()), /* prefix */ "");
  if (!current_rank_local_is_valid) { return input; }
  const auto& output = JUST(boxing_interpreter->Interpret(input, in_nd_sbp, out_nd_sbp,
                                                          in_parallel_desc, out_parallel_desc));
  return output;
}

auto* GetBoxingOutput =
    DECORATE(DECORATE(&CalcBoxingOutput, CheckGlobalTensorMeta), DisableRecusiveBoxingCall);

Maybe<void> Interpret(const UserOpExpr& user_op_expr, const TensorTuple& inputs,
                      TensorTuple* outputs, const OpExprInterpContext& ctx) {
  CHECK_EQ_OR_RETURN(outputs->size(), user_op_expr.output_size());
  // 这里获取的是第一个非host的input tensor的parallel_desc
  auto parallel_desc = JUST(GetParallelDesc(inputs, ctx, user_op_expr));
  // 需求是：提前检测check inputs tensor的的parallel
  // desc，如果发现不一致，也不报错，而是通过parallel_desc找到inputs里最大的rank数
  // 将其他tensor通过boxing（GetBoxingOutput()）拿到to_global后的tensor
  std::shared_ptr<const GlobalTensorInferResult> result;
  NonRecursiveMetaInfoConsistencyCheckScope scope;
  bool is_identical = true;
  int64_t max_rank_tensor_id = -1;
  int64_t max_rank = -1;
  vm::EagerBlobObjectList boxing_input_eager_blob_objects(inputs.size());
  // extand lifetime of boxing outputs to the end of this function
  TensorTuple boxing_inputs;
  if (inputs.empty()) {
    // check consistency placement and nd_sbp, do not check in non-src op because it is assumed that
    // InferSbp in op is a deterministic algorithm
    JUST(MetaInfoConsistencyCheck(parallel_desc, ctx.nd_sbp, 1, /* force_check */ false));
    const auto& infer_args =
        JUST(SrcOpGlobalTensorMetaInferArgs::New(ctx.attrs, parallel_desc, JUST(ctx.nd_sbp)));
    result = JUST(user_op_expr.mut_global_tensor_infer_cache()->GetOrInfer(*infer_args));
  } else {
    // inputs非空时，通过ctx.attrs和inputs New一个infer_args(GlobalTensorMetaInferArgs)
    // 再通过infer_args进行infer，并将infer的result放入cache中。
    // infer时首先做了2个check:
    //  - 1.CheckInputParallelDescIdentical（check inputs parallel desc是否一致，不一致则报错)
    //  - 2.CheckIsDeviceSupportedByOp
    // 然后主要对output tensors的TensorMeta进行了推导
    for (int i = 0; i < outputs->size(); ++i) {
      if ((*outputs)[i]) {
        const auto& nd_sbp = JUST((*outputs)[i]->nd_sbp());
        JUST((*outputs)[i]->set_consumer_nd_sbp_constraint(nd_sbp));
      }
    }

    const auto& infer_args = JUST(GlobalTensorMetaInferArgs::New(ctx.attrs, inputs));
    is_identical = JUST(IsInputParallelDescIdentical(*infer_args, user_op_expr));
    printf("\nis_identical >>>>>>>>>>>> %d\n", is_identical);
    if (!is_identical) {
      max_rank_tensor_id = JUST(GetMaxRankTensorId(inputs));
      max_rank = JUST(GetMaxRankId(JUST(inputs[max_rank_tensor_id]->parallel_desc())));
      printf("\nmax rank tensor id:%d  max rank >>>>>>>>>> ********* %d;\n",
             int(max_rank_tensor_id), int(max_rank));

      parallel_desc = JUST(inputs[max_rank_tensor_id]->parallel_desc());
      JUST(inputs[max_rank_tensor_id]->nd_sbp());
      Optional<int64_t> max_parallel_id;
      JUST(GetTensorDevice4CurrentProcessCtx(parallel_desc, &max_parallel_id));
      for (int i = 0; i < inputs.size(); ++i) {
        std::shared_ptr<Tensor> input = inputs.at(i);
        std::shared_ptr<Tensor> final_input = nullptr;
        Optional<int64_t> parallel_id;

        JUST(GetTensorDevice4CurrentProcessCtx(JUST(input->parallel_desc()), &parallel_id));
        printf("\nBefore GetBoxingOutput of input :%d; inputs size:%d; parallel_desc(origin:%s; "
               "target:%s)\n",
               i, int(inputs.size()),
               JUST(PlacementToString(JUST(input->parallel_desc())))->c_str(),
               JUST(PlacementToString(parallel_desc))->c_str());
        final_input = JUST(GetBoxingOutput(input, JUST(inputs[i]->nd_sbp()), parallel_desc,
                                           parallel_id.has_value() || max_parallel_id.has_value()));
        printf("\nAfter GetBoxingOutput of input :%d; parallel_desc:%s;\n", i,
               JUST(PlacementToString(JUST(input->parallel_desc())))->c_str());

        boxing_inputs.emplace_back(final_input);
        const auto& local_tensor = JUST(final_input->cur_rank_phy_tensor());
        boxing_input_eager_blob_objects.at(i) = JUST(local_tensor->eager_blob_object());
      }
      const auto& new_infer_args = JUST(GlobalTensorMetaInferArgs::New(ctx.attrs, boxing_inputs));
      printf("\n==============GetOrInfer 1===============\n");
      result = JUST(user_op_expr.mut_global_tensor_infer_cache()->GetOrInfer(*new_infer_args));
    } else {
      printf("\n==============GetOrInfer 2===============\n");
      result = JUST(user_op_expr.mut_global_tensor_infer_cache()->GetOrInfer(*infer_args));
    }
  }
  // 这里根据上面infer得到的output tensors的TensorMeta，然后New出最终的output tensors
  const auto& output_tensor_metas = result->output_tensor_metas();
  Optional<int64_t> parallel_id;
  const auto& tensor_device = JUST(GetTensorDevice4CurrentProcessCtx(parallel_desc, &parallel_id));
  for (int i = 0; i < outputs->size(); ++i) {
    if (!outputs->at(i)) {
      const auto& tensor_impl = JUST(EagerGlobalTensorImpl::New(
          output_tensor_metas[i], tensor_device, parallel_id, false, false));
      (*outputs)[i].reset(new GlobalTensor(tensor_impl));
    } else {
      JUST((*outputs)[i]->set_consumer_nd_sbp_constraint(NullOpt));
    }
  }
  // Do nothing if output_tensors has 0-size shape. Since the input of some ops is 0-size but the
  // output is not 0-size, it cannot be judged based on the input, such as flow.cat
  if (unlikely(JUST(CachedIsAllZeroSizeTensorMeta(output_tensor_metas)))) {
    return Maybe<void>::Ok();
  }
  // Run instruction Call
  const auto& kernel = JUST(user_op_expr.MutKernel4Stream(result->stream()));
  CHECK_EQ_OR_RETURN(kernel->output_tuple_indexes4mut2_obns().size(), 0)
      << Error::UnimplementedError() << GetDynamicOpGlobalFailedDebugString(user_op_expr, *kernel);
  // 遍历input tensors，对一些parallel_desc不适合的tensor，通过boxing来得到可用的tensor
  // vm::EagerBlobObjectList input_eager_blob_objects(boxing_inputs.size());
  // extand lifetime of boxing outputs to the end of this function
  TensorTuple boxing_outputs;
  for (int i = 0; i < boxing_inputs.size(); ++i) {
    std::shared_ptr<Tensor> input = boxing_inputs.at(i);
    const auto& infered_input_meta = result->input_tensor_metas().at(i);
    const auto& input_parallel_desc = JUST(input->parallel_desc());
    CHECK_OR_RETURN(input_parallel_desc == infered_input_meta->parallel_desc());
    bool is_host_input = user_op_expr.IsHostMemoryInput(i);
    Symbol<ParallelDesc> dst_parallel_desc =
        is_host_input
            ? JUST(ReplaceDeviceType(infered_input_meta->parallel_desc(), DeviceType::kCPU))
            : infered_input_meta->parallel_desc();
    if ((input_parallel_desc->parallel_num() != 1
         && infered_input_meta->nd_sbp() != JUST(input->nd_sbp()))
        || input_parallel_desc->device_type() != dst_parallel_desc->device_type()) {
      input = JUST(GetBoxingOutput(input, infered_input_meta->nd_sbp(), dst_parallel_desc,
                                   parallel_id.has_value()));
      boxing_outputs.emplace_back(input);
    }
    const auto& local_tensor = JUST(input->cur_rank_phy_tensor());
    boxing_input_eager_blob_objects.at(i) = JUST(local_tensor->eager_blob_object());
  }
  // Do nothing if the `parallel_desc` doesn't cover current ProcessCtx.
  if (!parallel_id.has_value()) { return Maybe<void>::Ok(); }
  vm::EagerBlobObjectList output_eager_blob_objects(outputs->size());
  for (int i = 0; i < outputs->size(); ++i) {
    const auto& local_tensor = JUST(outputs->at(i)->cur_rank_phy_tensor());
    output_eager_blob_objects.at(i) = JUST(local_tensor->eager_blob_object());
  }

  // dispatch op执行指令至虚拟机，执行该op/kernel
  JUST(PhysicalRun([&](InstructionsBuilder* builder) -> Maybe<void> {
    return builder->Call(kernel, std::move(boxing_input_eager_blob_objects),
                         std::move(output_eager_blob_objects), result, ctx, result->stream());
  }));
  return Maybe<void>::Ok();
}

auto* InterpretThenInitGlobalId = DECORATE(&Interpret, NonRecursiveInitGlobalId);

}  // namespace

Maybe<void> EagerGlobalInterpreter::ApplyImpl(const UserOpExpr& op_expr, const TensorTuple& inputs,
                                              TensorTuple* outputs,
                                              const OpExprInterpContext& ctx) const {
  return InterpretThenInitGlobalId(op_expr, inputs, outputs, ctx);
}

Maybe<void> EagerGlobalInterpreter::ApplyImpl(const VariableOpExpr& op_expr,
                                              const TensorTuple& inputs, TensorTuple* outputs,
                                              const OpExprInterpContext& ctx) const {
  OF_UNIMPLEMENTED();
}

namespace {

static constexpr auto* RecursiveGetBoxingOutput =
    DECORATE(&CalcBoxingOutput, CheckGlobalTensorMeta);

Maybe<void> RawGlobalToGlobal(const GlobalToGlobalOpExpr& op_expr, const TensorTuple& inputs,
                              TensorTuple* outputs, const OpExprInterpContext& ctx) {
  CHECK_EQ_OR_RETURN(inputs.size(), 1);
  CHECK_EQ_OR_RETURN(outputs->size(), 1);
  const auto& input = inputs.at(0);
  CHECK_OR_RETURN(input->is_global());  // NOLINT
  CHECK_OR_RETURN(ctx.parallel_desc.has_value());
  CHECK_OR_RETURN(ctx.nd_sbp.has_value());
  const auto& in_parallel_desc = JUST(input->parallel_desc());
  const auto& out_nd_sbp = JUST(ctx.nd_sbp);
  const auto& out_parallel_desc = JUST(ctx.parallel_desc);
  const auto& in_parallel_id = JUST(GetParallelId4CurrentProcessCtx(in_parallel_desc));
  const auto& out_parallel_id = JUST(GetParallelId4CurrentProcessCtx(out_parallel_desc));
  const auto& tensor =
      JUST(RecursiveGetBoxingOutput(input, out_nd_sbp, out_parallel_desc,
                                    in_parallel_id->has_value() || out_parallel_id->has_value()));
  CHECK_OR_RETURN(tensor);
  if (out_parallel_id->has_value()) {
    const auto& nd_sbp = JUST(tensor->nd_sbp());
    const auto& parallel_desc = JUST(tensor->parallel_desc());
    CHECK_OR_RETURN(nd_sbp == out_nd_sbp)
        << ". nd_sbp: " << NdSbpToString(nd_sbp) << ", out_nd_sbp" << NdSbpToString(out_nd_sbp);
    CHECK_OR_RETURN(parallel_desc == out_parallel_desc);
    outputs->at(0) = tensor;
  } else {
    GlobalTensorMeta tensor_meta(*tensor->shape(), tensor->dtype()->data_type(), out_nd_sbp,
                                 out_parallel_desc);
    const auto& tensor_impl =
        JUST(EagerGlobalTensorImpl::New(SymbolOf(tensor_meta), tensor->requires_grad(), false));
    (*outputs)[0].reset(new GlobalTensor(tensor_impl));
  }
  CHECK_OR_RETURN(outputs->at(0));
  return Maybe<void>::Ok();
}

static constexpr auto* GlobalToGlobal = DECORATE(&RawGlobalToGlobal, NonRecursiveInitGlobalId);

}  // namespace

Maybe<void> EagerGlobalInterpreter::ApplyImpl(const GlobalToGlobalOpExpr& op_expr,
                                              const TensorTuple& inputs, TensorTuple* outputs,
                                              const OpExprInterpContext& ctx) const {
  JUST(GlobalToGlobal(op_expr, inputs, outputs, ctx));
  return Maybe<void>::Ok();
}

Maybe<void> EagerGlobalInterpreter::ApplyImpl(const LocalToGlobalOpExpr& op_expr,
                                              const TensorTuple& inputs, TensorTuple* outputs,
                                              const OpExprInterpContext& ctx) const {
  OF_UNIMPLEMENTED();
}

Maybe<void> EagerGlobalInterpreter::ApplyImpl(const GlobalToLocalOpExpr& op_expr,
                                              const TensorTuple& inputs, TensorTuple* outputs,
                                              const OpExprInterpContext& ctx) const {
  CHECK_EQ_OR_RETURN(inputs.size(), 1);
  const auto& input_tensor = inputs.at(0);
  const auto& local_tensor = JUST(JUST(input_tensor->cur_rank_phy_tensor())->detach());
  bool requires_grad = autograd::GradMode::is_enabled() && input_tensor->requires_grad();
  JUST(local_tensor->set_requires_grad(requires_grad));
  local_tensor->set_is_leaf(!requires_grad);
  (*outputs)[0] = local_tensor;
  return Maybe<void>::Ok();
}

Maybe<void> EagerGlobalInterpreter::ApplyImpl(const CastToLocalOpExpr& op_expr,
                                              const TensorTuple& inputs, TensorTuple* outputs,
                                              const OpExprInterpContext& ctx) const {
  OF_UNIMPLEMENTED();
}

Maybe<void> EagerGlobalInterpreter::ApplyImpl(const CastFromLocalOpExpr& op_expr,
                                              const TensorTuple& inputs, TensorTuple* outputs,
                                              const OpExprInterpContext& ctx) const {
  OF_UNIMPLEMENTED();
}

Maybe<void> EagerGlobalInterpreter::ApplyImpl(const DistributeSplitOpExpr& op_expr,
                                              const TensorTuple& inputs, TensorTuple* outputs,
                                              const OpExprInterpContext& ctx) const {
  OF_UNIMPLEMENTED();
}

Maybe<void> EagerGlobalInterpreter::ApplyImpl(const DistributeCloneOpExpr& op_expr,
                                              const TensorTuple& inputs, TensorTuple* outputs,
                                              const OpExprInterpContext& ctx) const {
  OF_UNIMPLEMENTED();
}

Maybe<void> EagerGlobalInterpreter::ApplyImpl(const DistributeConcatOpExpr& op_expr,
                                              const TensorTuple& inputs, TensorTuple* outputs,
                                              const OpExprInterpContext& ctx) const {
  OF_UNIMPLEMENTED();
}

Maybe<void> EagerGlobalInterpreter::ApplyImpl(const DistributeAddOpExpr& op_expr,
                                              const TensorTuple& inputs, TensorTuple* outputs,
                                              const OpExprInterpContext& ctx) const {
  OF_UNIMPLEMENTED();
}

Maybe<void> EagerGlobalInterpreter::ApplyImpl(const SelectTopNOpExpr& op_expr,
                                              const TensorTuple& inputs, TensorTuple* outputs,
                                              const OpExprInterpContext& ctx) const {
  OF_UNIMPLEMENTED();
}

}  // namespace one
}  // namespace oneflow
