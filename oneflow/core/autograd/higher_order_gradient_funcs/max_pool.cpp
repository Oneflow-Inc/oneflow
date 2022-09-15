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

#include "oneflow/core/framework/op_expr_grad_function.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/common/container_util.h"

namespace oneflow {
namespace one {

struct MaxPoolGradGradCaptureState : public AutoGradCaptureState {
  bool grad_requires_grad = false;
  bool input_requires_grad = false;
};

template<int ndims>
class MaxPoolNdGradGrad : public OpExprGradFunction<MaxPoolGradGradCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override { return Maybe<void>::Ok(); }

  Maybe<void> Capture(MaxPoolGradGradCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    // dy, x, indice
    CHECK_EQ_OR_RETURN(inputs.size(), 3);   // NOLINT(maybe-need-error-msg)
    CHECK_EQ_OR_RETURN(outputs.size(), 1);  // NOLINT(maybe-need-error-msg)

    ctx->grad_requires_grad = inputs[0]->requires_grad();
    ctx->input_requires_grad = inputs[1]->requires_grad();
    if (ctx->grad_requires_grad) { ctx->SaveTensorForBackward(inputs[2]); }

    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const MaxPoolGradGradCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)
    in_grads->resize(3);

    if (ctx->grad_requires_grad) {
      const auto& indices = JUST(VectorAt(ctx->SavedTensors(), 0));
      (*in_grads)[0] = JUST(functional::MaxPoolNdGradGrad(out_grads[0], indices, ndims));
    }
    if (ctx->input_requires_grad) { (*in_grads)[1] = JUST(functional::ZerosLike(out_grads[0])); }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("max_pool_1d_grad", MaxPoolNdGradGrad<1>);
REGISTER_OP_EXPR_GRAD_FUNCTION("max_pool_2d_grad", MaxPoolNdGradGrad<2>);
REGISTER_OP_EXPR_GRAD_FUNCTION("max_pool_3d_grad", MaxPoolNdGradGrad<3>);
// REGISTER_OP_EXPR_GRAD_FUNCTION("adaptive_max_pool1d_grad", MaxPoolNdGradGrad<1>);
// REGISTER_OP_EXPR_GRAD_FUNCTION("adaptive_max_pool2d_grad", MaxPoolNdGradGrad<2>);
// REGISTER_OP_EXPR_GRAD_FUNCTION("adaptive_max_pool3d_grad", MaxPoolNdGradGrad<3>);
}  // namespace one
}  // namespace oneflow
