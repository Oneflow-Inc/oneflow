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

#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/scalar.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_expr_grad_function.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/functional/functional_api.yaml.h"

namespace oneflow {
namespace one {

struct FusedMSAAttentionCaptureState : public AutoGradCaptureState {
  bool input_requires_grad = false;
  bool bias_requires_grad = false;
  int32_t input_size = 3;
  std::string mode = "row";
  float scale = 1.0;
};

class FusedMSAAttention : public OpExprGradFunction<FusedMSAAttentionCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(FusedMSAAttentionCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const FusedMSAAttentionCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
};

Maybe<void> FusedMSAAttention::Init(const OpExpr& op) {
  const UserOpExpr* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  return Maybe<void>::Ok();
}

Maybe<void> FusedMSAAttention::Capture(FusedMSAAttentionCaptureState* ctx,
                                       const TensorTuple& inputs, const TensorTuple& outputs,
                                       const AttrMap& attrs) const {
  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  const std::string& mode = JUST(composed_attrs.GetAttr<std::string>("mode"));
  if (mode == "row" || mode == "triangle_start" || mode == "triangle_end") {
    CHECK_EQ_OR_RETURN(inputs.size(), 3);
    ctx->bias_requires_grad = inputs.at(2)->requires_grad();
  } else {
    CHECK_EQ_OR_RETURN(inputs.size(), 2);
    ctx->input_size = 2;
  }
  ctx->mode = mode;
  ctx->input_requires_grad = inputs.at(0)->requires_grad();
  if (!ctx->input_requires_grad) { return Maybe<void>::Ok(); }

  ctx->scale = JUST(composed_attrs.GetAttr<float>("scale"));

  ctx->SaveTensorForBackward(outputs.at(0));  // y
  return Maybe<void>::Ok();
}

Maybe<void> FusedMSAAttention::Apply(const FusedMSAAttentionCaptureState* ctx,
                                     const TensorTuple& out_grads, TensorTuple* in_grads) const {
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // dy
  if (!ctx->input_requires_grad && !ctx->bias_requires_grad) { return Maybe<void>::Ok(); }
  in_grads->resize(ctx->input_size);

  const std::shared_ptr<oneflow::one::Tensor>& y = ctx->SavedTensors().at(0);
  const std::shared_ptr<oneflow::one::Tensor>& input_grad =
      JUST(functional::FusedMSAAttentionGrad(y, out_grads.at(0), ctx->scale, ctx->mode));

  in_grads->at(0) = input_grad;
  if (ctx->bias_requires_grad) {
    in_grads->at(2) = JUST(functional::ScalarMul(
        1 / ctx->scale,
        JUST(functional::ReduceSum(input_grad, {0}, true))));  // pair_grad: B,h,S,S -> 1, h, S, S
  }

  return Maybe<void>::Ok();
}

struct FusedMSASigmoidMulCaptureState : public AutoGradCaptureState {
  bool input_requires_grad = false;
};

class FusedMSASigmoidMul : public OpExprGradFunction<FusedMSASigmoidMulCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const UserOpExpr* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);
    base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
    return Maybe<void>::Ok();
  }
  Maybe<void> Capture(FusedMSASigmoidMulCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    bool inplace = JUST(composed_attrs.GetAttr<bool>("inplace"));
    CHECK_EQ_OR_RETURN(inplace, false);
    ctx->input_requires_grad = inputs.at(0)->requires_grad();
    ctx->SaveTensorForBackward(inputs.at(0));
    ctx->SaveTensorForBackward(inputs.at(1));
    return Maybe<void>::Ok();
  }
  Maybe<void> Apply(const FusedMSASigmoidMulCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    if (!ctx->input_requires_grad) return Maybe<void>::Ok();
    in_grads->resize(2);
    const std::shared_ptr<oneflow::one::Tensor>& g = ctx->SavedTensors().at(0);
    const std::shared_ptr<oneflow::one::Tensor>& x = ctx->SavedTensors().at(1);
    const std::shared_ptr<oneflow::one::Tensor>& dgx =
        JUST(functional::FusedMSASigmoidMulGrad(out_grads.at(0), g, x, false));
    auto shape = dgx->shape();
    std::vector<int64_t> start, step, stop;
    int i = 0;
    for (; i < shape->size(); ++i) {
      start.push_back(0);
      stop.push_back(shape->At(i));
      step.push_back(1);
    }
    step[-1] = 2;
    in_grads->at(0) = JUST(functional::Slice(dgx, start, stop, step, true));
    start[-1] = 1;
    in_grads->at(1) = JUST(functional::Slice(dgx, start, stop, step, true));
    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

struct FusedMSADropoutAddCaptureState : public AutoGradCaptureState {
  bool input_requires_grad = false;
};

class FusedMSADropoutAdd : public OpExprGradFunction<FusedMSADropoutAddCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const UserOpExpr* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);
    base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
    return Maybe<void>::Ok();
  }
  Maybe<void> Capture(FusedMSADropoutAddCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    bool inplace = JUST(composed_attrs.GetAttr<bool>("inplace"));
    CHECK_EQ_OR_RETURN(inplace, false);
    ctx->input_requires_grad = inputs.at(0)->requires_grad();
    ctx->SaveTensorForBackward(inputs.at(1));
    return Maybe<void>::Ok();
  }
  Maybe<void> Apply(const FusedMSADropoutAddCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    if (!ctx->input_requires_grad) return Maybe<void>::Ok();
    in_grads->resize(2);
    const std::shared_ptr<oneflow::one::Tensor>& mask = ctx->SavedTensors().at(0);
    const std::shared_ptr<oneflow::one::Tensor>& dx =
        JUST(functional::FusedMSADropoutAddGrad(out_grads.at(0), mask, false));
    in_grads->at(0) = dx;
    in_grads->at(1) = out_grads.at(0);
    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("fused_msa_attention", FusedMSAAttention);
REGISTER_OP_EXPR_GRAD_FUNCTION("fused_msa_sigmoid_mul", FusedMSASigmoidMul);
REGISTER_OP_EXPR_GRAD_FUNCTION("fused_msa_dropout_add", FusedMSADropoutAdd);

}  // namespace one
}  // namespace oneflow
