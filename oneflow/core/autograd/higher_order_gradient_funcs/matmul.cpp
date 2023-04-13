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
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/functional/functional_api.yaml.h"
#include "oneflow/core/functional/sequence_function.h"

namespace oneflow {
namespace one {

struct BroadcastMatmulGradBGradCaptureState : public AutoGradCaptureState {
  bool a_requires_grad = false;
  bool b_requires_grad = false;
  size_t a_index = 0;
  size_t b_index = 1;
  double alpha = 1.0;
};

class BroadcastMatmulGradBGrad : public OpExprGradFunction<BroadcastMatmulGradBGradCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const UserOpExpr* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
    base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
    return Maybe<void>::Ok();
  };
  Maybe<void> Capture(BroadcastMatmulGradBGradCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 2);   // NOLINT(maybe-need-error-msg)
    CHECK_EQ_OR_RETURN(outputs.size(), 1);  // NOLINT(maybe-need-error-msg)

    ctx->a_requires_grad = inputs.at(0)->requires_grad();
    ctx->b_requires_grad = inputs.at(1)->requires_grad();
    if (ctx->a_requires_grad) { ctx->b_index = ctx->SaveTensorForBackward(inputs.at(1)); }
    if (ctx->b_requires_grad) { ctx->a_index = ctx->SaveTensorForBackward(inputs.at(0)); }

    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    ctx->alpha = JUST(composed_attrs.GetAttr<double>("alpha"));

    return Maybe<void>::Ok();
  }
  Maybe<void> Apply(const BroadcastMatmulGradBGradCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    in_grads->resize(2);

    // for matmul: input_a[dims..., m, k] * input_b[k, n] -> [dims..., m, n]
    // if forward: BroadcastMatmulGradB(input_a, JUST(VectorAt(out_grads, 0)), ctx->alpha))
    //       then: a.shape = [dims..., m, k], b.shape = [dims..., m, n], grad.shape = [k, n]
    // if forward: BroadcastMatmulGradB(JUST(VectorAt(out_grads, 0)), input_a, ctx->alpha))
    //       then: a.shape = [dims..., m, n], b.shape = [dims..., m, k], grad.shape = [n, k]
    if (ctx->a_requires_grad) {
      const auto& b = ctx->SavedTensors()[ctx->b_index];
      in_grads->at(0) = JUST(functional::MatMul(b, out_grads.at(0), false, true, ctx->alpha));
    }
    if (ctx->b_requires_grad) {
      const auto& a = ctx->SavedTensors()[ctx->a_index];
      in_grads->at(1) = JUST(functional::MatMul(a, out_grads.at(0), false, false, ctx->alpha));
    }

    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("broadcast_matmul_grad_b", BroadcastMatmulGradBGrad);

}  // namespace one
}  // namespace oneflow
