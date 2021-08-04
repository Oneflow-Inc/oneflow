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

#include "oneflow/core/autograd/autograd_engine.h"
#include "oneflow/core/autograd/autograd_mode.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/framework/op_arg_util.h"
#include "oneflow/core/framework/op_expr_grad_function.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"

namespace oneflow {
namespace one {

Maybe<void> LazyInterpreter::Apply(const OpExpr& op_expr, const TensorTuple& inputs,
                                   TensorTuple* outputs, const OpExprInterpContext& ctx) const {
#define APPLY_IF(op_type)                                              \
  if (const auto* op = dynamic_cast<const op_type##Expr*>(&op_expr)) { \
    return ApplyImpl(*op, inputs, outputs, ctx);                       \
  }

  APPLY_IF(FeedInputOp);
  APPLY_IF(FeedVariableOp);
  APPLY_IF(FetchOutputOp);
  APPLY_IF(UserOp);
  APPLY_IF(FunctionOp);
#undef APPLY_IF

  OF_UNIMPLEMENTED() << "The type " << op_expr.op_type_name()
                     << " has not been supported in LazyInterpreter::Apply.";
}

Maybe<void> EagerInterpreter::Apply(const OpExpr& op_expr, const TensorTuple& inputs,
                                    TensorTuple* outputs, const OpExprInterpContext& ctx) const {
#define APPLY_IF(op_type)                                              \
  if (const auto* op = dynamic_cast<const op_type##Expr*>(&op_expr)) { \
    return ApplyImpl(*op, inputs, outputs, ctx);                       \
  }

  APPLY_IF(UserOp);
  APPLY_IF(VariableOp);
  APPLY_IF(CastToMirroredOp);
  APPLY_IF(CastFromMirroredOp);
  APPLY_IF(CastToConsistentOp);
  APPLY_IF(CastFromConsistentOp);
  APPLY_IF(DistributeSplitOp);
  APPLY_IF(DistributeCloneOp);
  APPLY_IF(DistributeConcatOp);
  APPLY_IF(DistributeAddOp);
  APPLY_IF(FunctionOp);
  APPLY_IF(SelectFirstOp)
#undef APPLY_IF

  OF_UNIMPLEMENTED() << "The type " << op_expr.op_type_name()
                     << " has not been supported in EagerInterpreter::Apply.";
}

Maybe<void> EagerInterpreter::ApplyImpl(const FunctionOpExpr& op_expr, const TensorTuple& inputs,
                                        TensorTuple* outputs,
                                        const OpExprInterpContext& ctx) const {
  // TODO(hjchen2)
  UNIMPLEMENTED();
  return Maybe<void>::Ok();
}

namespace {

Maybe<void> DetermineIsLeaf(TensorTuple* outputs, const bool& is_leaf, const bool& requires_grad) {
  bool logical_is_leaf = is_leaf || !requires_grad;
  for (auto& output : *outputs) { output->set_is_leaf(logical_is_leaf); }
  return Maybe<void>::Ok();
}

Maybe<void> DetermineRequiresGrad(TensorTuple* outputs, const bool& requires_grad) {
  for (auto& output : *outputs) { output->set_requires_grad(requires_grad); }
  return Maybe<void>::Ok();
}

}  // namespace

Maybe<void> AutogradInterpreter::Apply(const OpExpr& op_expr, const TensorTuple& inputs,
                                       TensorTuple* outputs, const OpExprInterpContext& ctx) const {
  bool requires_grad = false;
  if (autograd::GradMode::is_enabled() && !JUST(op_expr.IsGradDisabled())) {
    requires_grad =
        std::any_of(inputs.begin(), inputs.end(),
                    [](const std::shared_ptr<Tensor>& tensor) { return tensor->requires_grad(); });
  }
  {
    autograd::AutoGradMode mode(false);
    JUST(internal_->Apply(op_expr, inputs, outputs, ctx));
    JUST(DetermineIsLeaf(outputs, inputs.size() == 0, requires_grad));
    JUST(DetermineRequiresGrad(outputs, requires_grad));
  }
  if (requires_grad) {
    const auto& grad_closure = JUST(op_expr.GetOrCreateOpGradClosure());
    JUST(grad_closure->Capture(inputs, *outputs, ctx));

    auto backward_fn =
        std::make_shared<std::function<Maybe<void>(const TensorTuple&, TensorTuple*, bool)>>(
            [=](const TensorTuple& out_grads, TensorTuple* in_grads,
                bool create_graph) -> Maybe<void> {
              autograd::AutoGradMode mode(create_graph);
              JUST(grad_closure->Apply(out_grads, in_grads));
              return Maybe<void>::Ok();
            });
    JUST(GetThreadLocalAutogradEngine()->AddBackwardFuncPtr(op_expr.op_type_name() + "_backward",
                                                            backward_fn, inputs, outputs));
  }
  return Maybe<void>::Ok();
}

}  // namespace one
}  // namespace oneflow
