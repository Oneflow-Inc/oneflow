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
#include "oneflow/core/framework/op_expr_grad_function.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/job/lazy_mode.h"
#include "oneflow/core/profiler/profiler.h"

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
  APPLY_IF(GlobalToGlobalOp);
  APPLY_IF(FunctionOp);
  APPLY_IF(ImageDecoderRandomCropResizeOp);
#undef APPLY_IF

  OF_UNIMPLEMENTED() << "The type " << op_expr.op_type_name()
                     << " has not been supported in LazyInterpreter::Apply.";
}

Maybe<void> EagerInterpreter::Apply(const OpExpr& op_expr, const TensorTuple& inputs,
                                    TensorTuple* outputs, const OpExprInterpContext& ctx) const {
  // In the op interpreter, judge whether to open the global mode to avoid recursion caused by
  // GlobalMode.
  // The global mode is enabled only if it was enabled and the current operation is a local
  // operation.
  auto global_mode_gurad = GlobalMode::Guard(GlobalMode::is_enabled() && is_local_,
                                             GlobalMode::nd_sbp(), GlobalMode::parallel_desc());

#define APPLY_IF(op_type)                                              \
  if (const auto* op = dynamic_cast<const op_type##Expr*>(&op_expr)) { \
    return ApplyImpl(*op, inputs, outputs, ctx);                       \
  }

  APPLY_IF(UserOp);
  APPLY_IF(VariableOp);
  APPLY_IF(CastToLocalOp);
  APPLY_IF(CastFromLocalOp);
  APPLY_IF(GlobalToGlobalOp);
  APPLY_IF(LocalToGlobalOp);
  APPLY_IF(GlobalToLocalOp);
  APPLY_IF(DistributeSplitOp);
  APPLY_IF(DistributeCloneOp);
  APPLY_IF(DistributeConcatOp);
  APPLY_IF(DistributeAddOp);
  APPLY_IF(FunctionOp);
  APPLY_IF(SelectTopNOp)
#undef APPLY_IF

  OF_UNIMPLEMENTED() << "The type " << op_expr.op_type_name()
                     << " has not been supported in EagerInterpreter::Apply.";
}

Maybe<void> EagerInterpreter::ApplyImpl(const FunctionOpExpr& op_expr, const TensorTuple& inputs,
                                        TensorTuple* outputs, const OpExprInterpContext&) const {
  // Must reset ctx in each forward
  op_expr.reset_state();
  std::shared_ptr<FunctionAutoGradCaptureState> ctx = op_expr.state();
  *outputs = *(op_expr.forward()(ctx, inputs));
  return Maybe<void>::Ok();
}

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
  }
  // Lazy mode will construct backward compute graph in passes, so disable autograd if lazy mode.
  std::shared_ptr<OpExprGradClosure> grad_closure(nullptr);
  if (requires_grad) {
    OF_PROFILER_RANGE_PUSH("autograd.GetOrCreateOpGradClosure");
    grad_closure = JUST(op_expr.GetOrCreateOpGradClosure());
    auto backward_fn = std::make_shared<BackwardFunction>();
    backward_fn->body = [=](const TensorTuple& out_grads, TensorTuple* in_grads,
                            bool create_graph) -> Maybe<void> {
      autograd::AutoGradMode mode(create_graph);
      JUST(grad_closure->Apply(out_grads, in_grads));
      return Maybe<void>::Ok();
    };
    backward_fn->status = [=]() { return grad_closure->state()->SavedTensors().size() > 0; };
    OF_PROFILER_RANGE_POP();
    OF_PROFILER_RANGE_PUSH("autograd.AddNode");
    JUST(GetThreadLocalAutogradEngine()->AddNode(op_expr.op_type_name() + "_backward", backward_fn,
                                                 inputs, outputs));
    OF_PROFILER_RANGE_POP();
  }

  if (requires_grad) {
    OF_PROFILER_RANGE_GUARD("autograd.Capture");
    // Capture inputs and outputs after `AddNode` because of that grad function
    // node has been attached to them.
    JUST(grad_closure->Capture(inputs, *outputs, ctx));
  }
  // Update outputs autograd meta
  // Note: if requires_grad is True, we will create a new autograd meta for each output
  // in `AddNode` to support inplace operation, so the update should after
  // `AddNode`
  for (auto& output : *outputs) {
    output->set_is_leaf(inputs.size() == 0 || !requires_grad);
    // If the output `requires_grad` is true, it means that the output is inplaced.
    // The output `requires_grad` should be determined by this:
    //   - If the inplaced output `requires_grad` is true, then the autograd must be disabled,
    //     so the output `requires_grad` should never be changed.
    //   - If the inplaced output `requires_grad` is false, then the output `requires_grad`
    //     shoule be inferred by autograd mode and inputs. For example,
    //
    //     >>> import oneflow as flow
    //     >>> x = flow.ones(4, 4, requires_grad=False)
    //     >>> y = flow.ones(4, 4, requires_grad=True)
    //     >>> x += y
    //     >>> x.requires_grad
    //     True
    //     >>> with flow.no_grad():
    //     >>>    x += y
    //     >>> x.requires_grad
    //     False
    //
    //   - If there is no inplace, the output `requires_grad` should be inferred by autograd
    //     mode and inputs.
    if (!output->requires_grad()) {
      JUST(output->set_requires_grad(
          requires_grad && IsSupportRequireGradDataType(output->dtype()->data_type())));
    }
  }
  return Maybe<void>::Ok();
}

}  // namespace one
}  // namespace oneflow
