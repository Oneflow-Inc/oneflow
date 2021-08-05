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
#include "oneflow/core/framework/op_arg_util.h"
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
#include "oneflow/user/kernels/stateful_local_opkernel.h"

namespace oneflow {
namespace one {

namespace {

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

}  // namespace

Maybe<void> Interpret(const UserOpExpr& user_op_expr, const TensorTuple& inputs,
                      TensorTuple* outputs, const OpExprInterpContext& ctx) {
  CHECK_EQ_OR_RETURN(outputs->size(), user_op_expr.output_size());
  const auto& parallel_desc = JUST(GetParallelDesc(inputs, ctx));
  std::shared_ptr<const ConsistentTensorInferResult> result;
  if (inputs.empty()) {
    const auto& infer_args = JUST(SrcOpConsistentTensorMetaInferArgs::New(
        ctx.attrs, parallel_desc, JUST(ctx.parallel_distribution.value())));
    result = JUST(user_op_expr.mut_consistent_tensor_infer_cache()->GetOrInfer(*infer_args));
  } else {
    const auto& infer_args = JUST(ConsistentTensorMetaInferArgs::New(ctx.attrs, inputs));
    result = JUST(user_op_expr.mut_consistent_tensor_infer_cache()->GetOrInfer(*infer_args));
  }
  const auto& output_tensor_metas = result->output_tensor_metas();
  Optional<int64_t> parallel_id;
  const auto& device = JUST(GetDevice4CurrentProcessCtx(parallel_desc, &parallel_id));
  for (int i = 0; i < outputs->size(); ++i) {
    const auto& tensor_impl = JUST(EagerConsistentTensorImpl::New(output_tensor_metas.at(i), device,
                                                                  parallel_id, false, false));
    const auto& rpc_token = JUST(RpcToken::NewMetaRpcToken());
    JUST(tensor_impl->set_rpc_token(rpc_token));
    outputs->at(i).reset(new ConsistentTensor(tensor_impl));
  }
  // Do nothing if the `parallel_desc` doesn't cover current ProcessCtx.
  if (!device) { return Maybe<void>::Ok(); }
  // Run instruction LocalCallOpKernel
  const auto& kernel = JUST(user_op_expr.MutKernel4Device(*device));
  CHECK_EQ_OR_RETURN(kernel->output_tuple_indexes4mut2_obns().size(), 0)
      << Error::Unimplemented() << GetDynamicOpConsistentFailedDebugString(user_op_expr, *kernel);
  std::shared_ptr<EagerBlobObjectList> input_eager_blob_objects =
      std::make_shared<EagerBlobObjectList>(inputs.size());
  for (int i = 0; i < inputs.size(); ++i) {
    const auto& local_tensor = JUST(inputs.at(i)->cur_rank_phy_tensor());
    input_eager_blob_objects->at(i) = JUST(local_tensor->eager_blob_object());
  }
  std::shared_ptr<EagerBlobObjectList> output_eager_blob_objects =
      std::make_shared<EagerBlobObjectList>(outputs->size());
  for (int i = 0; i < outputs->size(); ++i) {
    const auto& local_tensor = JUST(outputs->at(i)->cur_rank_phy_tensor());
    output_eager_blob_objects->at(i) = JUST(local_tensor->eager_blob_object());
  }
  const auto& instr_type_name = JUST(GetLocalCallInstructionName(parallel_desc->device_tag()));
  JUST(PhysicalRun([&](InstructionsBuilder* builder) -> Maybe<void> {
    return builder->LocalCallOpKernel(kernel, input_eager_blob_objects, output_eager_blob_objects,
                                      ctx, parallel_desc.shared_from_symbol(), instr_type_name);
  }));
  return Maybe<void>::Ok();
}

Maybe<void> EagerConsistentInterpreter::ApplyImpl(const UserOpExpr& op_expr,
                                                  const TensorTuple& inputs, TensorTuple* outputs,
                                                  const OpExprInterpContext& ctx) const {
  return Interpret(op_expr, inputs, outputs, ctx);
}

Maybe<void> EagerConsistentInterpreter::ApplyImpl(const VariableOpExpr& op_expr,
                                                  const TensorTuple& inputs, TensorTuple* outputs,
                                                  const OpExprInterpContext& ctx) const {
  OF_UNIMPLEMENTED();
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

}  // namespace one
}  // namespace oneflow
