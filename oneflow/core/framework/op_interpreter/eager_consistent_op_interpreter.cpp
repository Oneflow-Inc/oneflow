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
#include "oneflow/core/job/placement_scope.h"
#include "oneflow/core/eager/foreign_boxing_util.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/autograd/autograd_mode.h"

namespace oneflow {
namespace one {

Maybe<void> Interpret(const UserOpExpr& user_op_expr, const TensorTuple& inputs,
                      TensorTuple* outputs, const AttrMap& attrs) {
  CHECK_EQ_OR_RETURN(outputs->size(), user_op_expr.output_size());
  ConsistentTensorMetaInferArgs infer_args{};
  const auto& placement_scope = JUST(GetCurrentScope())->placement_scope();
  JUST(infer_args.Init(inputs, placement_scope, attrs));
  const auto& resualt =
      JUST(user_op_expr.mut_consistent_tensor_infer_cache()->GetOrInfer(infer_args));
  const auto& output_tensor_metas = resualt->output_tensor_metas();
  const auto& parallel_desc =
      JUST(placement_scope->GetParallelDesc(user_op_expr.op_type_name())).shared_from_symbol();
  int64_t parallel_id = -1;
  const auto& device = JUST(parallel_desc->GetDevice4CurrentProcessCtx(&parallel_id));
  using TensorImpl = EagerConsistentTensorImpl;
  TensorImpl::NewMethod New =
      (device ? &TensorImpl::NewWithPhyTensor : &TensorImpl::NewWithoutPhyTensor);
  for (int i = 0; i < outputs->size(); ++i) {
    const auto& tensor_impl =
        JUST(New(output_tensor_metas.at(i), device, parallel_id, false, false));
    outputs->at(i).reset(new ConsistentTensor(tensor_impl));
  }
  // Do nothing if the `parallel_desc` doesn't cover current ProcessCtx.
  if (!device) { return Maybe<void>::Ok(); }
  // Run instruction LocalCallOpKernel
  const auto& kernel = JUST(user_op_expr.MutKernel4Device(*device));
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
                                      attrs, parallel_desc, instr_type_name);
  }));
  return Maybe<void>::Ok();
}

Maybe<void> EagerConsistentInterpreter::ApplyImpl(const UserOpExpr& op_expr,
                                                  const TensorTuple& inputs, TensorTuple* outputs,
                                                  const AttrMap& attrs) const {
  return Interpret(op_expr, inputs, outputs, attrs);
}

Maybe<void> EagerConsistentInterpreter::ApplyImpl(const VariableOpExpr& op_expr,
                                                  const TensorTuple& inputs, TensorTuple* outputs,
                                                  const AttrMap& attrs) const {
  OF_UNIMPLEMENTED();
}

Maybe<void> EagerConsistentInterpreter::ApplyImpl(const CastToMirroredOpExpr& op_expr,
                                                  const TensorTuple& inputs, TensorTuple* outputs,
                                                  const AttrMap& attrs) const {
  OF_UNIMPLEMENTED();
}

Maybe<void> EagerConsistentInterpreter::ApplyImpl(const CastFromMirroredOpExpr& op_expr,
                                                  const TensorTuple& inputs, TensorTuple* outputs,
                                                  const AttrMap& attrs) const {
  OF_UNIMPLEMENTED();
}

Maybe<void> EagerConsistentInterpreter::ApplyImpl(const CastToConsistentOpExpr& op_expr,
                                                  const TensorTuple& inputs, TensorTuple* outputs,
                                                  const AttrMap& attrs) const {
  CHECK_EQ_OR_RETURN(inputs.size(), 1);
  CHECK_OR_RETURN(!inputs.at(0)->is_consistent());
  const auto& input_tensor = JUST(inputs.at(0)->detach());
  const auto& input_mirrored_tensor = std::dynamic_pointer_cast<MirroredTensor>(input_tensor);
  CHECK_OR_RETURN(input_mirrored_tensor) << Error::ValueError("Tensor Cast Error");
  bool requires_grad = autograd::GradMode::is_enabled() && inputs.at(0)->requires_grad();
  input_mirrored_tensor->set_requires_grad(requires_grad);
  input_mirrored_tensor->set_is_leaf(!requires_grad);
  const auto& parallel_distribution = op_expr.parallel_distribution();
  const auto& parallel_desc = op_expr.parallel_desc();
  std::shared_ptr<EagerConsistentTensorImpl> eager_consistent_tensor_impl = JUST(
      EagerConsistentTensorImpl::New(input_mirrored_tensor, parallel_distribution, parallel_desc));
  std::shared_ptr<ConsistentTensor> consistent_tensor =
      std::make_shared<ConsistentTensor>(eager_consistent_tensor_impl);
  const auto& out_tensor = std::dynamic_pointer_cast<Tensor>(consistent_tensor);
  CHECK_OR_RETURN(out_tensor) << Error::ValueError("Tensor Cast Error");
  outputs->at(0) = out_tensor;
  return Maybe<void>::Ok();
}

Maybe<void> EagerConsistentInterpreter::ApplyImpl(const CastFromConsistentOpExpr& op_expr,
                                                  const TensorTuple& inputs, TensorTuple* outputs,
                                                  const AttrMap& attrs) const {
  CHECK_EQ_OR_RETURN(inputs.size(), 1);
  CHECK_OR_RETURN(inputs.at(0)->is_consistent());
  const auto& input_consistent_tensor = std::dynamic_pointer_cast<ConsistentTensor>(inputs.at(0));
  CHECK_OR_RETURN(input_consistent_tensor) << Error::ValueError("Tensor Cast Error");
  std::shared_ptr<EagerConsistentTensorImpl> eager_consistent_tensor_impl =
      std::dynamic_pointer_cast<EagerConsistentTensorImpl>(
          JUST(input_consistent_tensor->consistent_tensor_impl()));
  CHECK_OR_RETURN(eager_consistent_tensor_impl) << Error::ValueError("TensorImpl Cast Error");
  const std::shared_ptr<Tensor>& mirrored_tensor =
      JUST(JUST(eager_consistent_tensor_impl->cur_rank_phy_tensor())->detach());
  bool requires_grad = autograd::GradMode::is_enabled() && inputs.at(0)->requires_grad();
  mirrored_tensor->set_requires_grad(requires_grad);
  mirrored_tensor->set_is_leaf(!requires_grad);
  outputs->at(0) = mirrored_tensor;
  return Maybe<void>::Ok();
}

Maybe<void> EagerConsistentInterpreter::ApplyImpl(const DistributeSplitOpExpr& op_expr,
                                                  const TensorTuple& inputs, TensorTuple* outputs,
                                                  const AttrMap& attrs) const {
  OF_UNIMPLEMENTED();
}

Maybe<void> EagerConsistentInterpreter::ApplyImpl(const DistributeCloneOpExpr& op_expr,
                                                  const TensorTuple& inputs, TensorTuple* outputs,
                                                  const AttrMap& attrs) const {
  OF_UNIMPLEMENTED();
}

Maybe<void> EagerConsistentInterpreter::ApplyImpl(const DistributeConcatOpExpr& op_expr,
                                                  const TensorTuple& inputs, TensorTuple* outputs,
                                                  const AttrMap& attrs) const {
  OF_UNIMPLEMENTED();
}

Maybe<void> EagerConsistentInterpreter::ApplyImpl(const DistributeAddOpExpr& op_expr,
                                                  const TensorTuple& inputs, TensorTuple* outputs,
                                                  const AttrMap& attrs) const {
  OF_UNIMPLEMENTED();
}

}  // namespace one
}  // namespace oneflow
