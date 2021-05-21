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

namespace oneflow {
namespace one {

Maybe<void> EagerConsistentInterpreter::ApplyImpl(const UserOpExpr& op_expr,
                                                  const TensorTuple& inputs, TensorTuple* outputs,
                                                  const AttrMap& attrs) const {
  OF_UNIMPLEMENTED();
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
  if (inputs.at(0)->is_consistent()) {
    outputs->at(0) = inputs.at(0);
    return Maybe<void>::Ok();
  }
  const auto& input_mirrored_tensor = std::dynamic_pointer_cast<MirroredTensor>(inputs.at(0));
  CHECK_OR_RETURN(input_mirrored_tensor) << Error::ValueError("Tensor Cast Error");
  std::shared_ptr<EagerMirroredTensorImpl> eager_mirrored_tensor_impl =
      std::dynamic_pointer_cast<EagerMirroredTensorImpl>(
          JUST(input_mirrored_tensor->mirrored_tensor_impl()));
  CHECK_OR_RETURN(eager_mirrored_tensor_impl) << Error::ValueError("TensorImpl Cast Error");
  std::shared_ptr<const cfg::ParallelDistribution> parallel_distribution =
      JUST(op_expr.parallel_distribution());
  std::shared_ptr<const ParallelDesc> parallel_desc = JUST(op_expr.parallel_desc());
  std::shared_ptr<EagerConsistentTensorImpl> eager_consistent_tensor_impl =
      JUST(EagerConsistentTensorImpl::New(eager_mirrored_tensor_impl, parallel_distribution,
                                          parallel_desc));
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
  if (!inputs.at(0)->is_consistent()) {
    outputs->at(0) = inputs.at(0);
    return Maybe<void>::Ok();
  }
  const auto& input_consistent_tensor = std::dynamic_pointer_cast<ConsistentTensor>(inputs.at(0));
  CHECK_OR_RETURN(input_consistent_tensor) << Error::ValueError("Tensor Cast Error");
  std::shared_ptr<EagerConsistentTensorImpl> eager_consistent_tensor_impl =
      std::dynamic_pointer_cast<EagerConsistentTensorImpl>(
          JUST(input_consistent_tensor->consistent_tensor_impl()));
  CHECK_OR_RETURN(eager_consistent_tensor_impl) << Error::ValueError("TensorImpl Cast Error");
  std::shared_ptr<EagerMirroredTensorImpl> eager_mirrored_tensor_impl =
      eager_consistent_tensor_impl->cur_rank_phy_tensor_impl();
  std::shared_ptr<MirroredTensor> mirrored_tensor =
      std::make_shared<MirroredTensor>(eager_mirrored_tensor_impl);
  const auto& out_tensor = std::dynamic_pointer_cast<Tensor>(mirrored_tensor);
  CHECK_OR_RETURN(out_tensor) << Error::ValueError("Tensor Cast Error");
  outputs->at(0) = out_tensor;
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
