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

struct FusedMSASoftmaxCaptureState : public AutoGradCaptureState {
  bool input_requires_grad = false;
  bool bias_requires_grad = false;
  int32_t input_size = 3;
  std::string mode = "row";
  float scale = 1.0;
  bool inplace = false;
};

class FusedMSASoftmax : public OpExprGradFunction<FusedMSASoftmaxCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(FusedMSASoftmaxCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const FusedMSASoftmaxCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
};

Maybe<void> FusedMSASoftmax::Init(const OpExpr& op) {
  const UserOpExpr* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  return Maybe<void>::Ok();
}

Maybe<void> FusedMSASoftmax::Capture(FusedMSASoftmaxCaptureState* ctx, const TensorTuple& inputs,
                                     const TensorTuple& outputs, const AttrMap& attrs) const {
  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  const std::string& mode = JUST(composed_attrs.GetAttr<std::string>("mode"));
  const bool& inplace = JUST(composed_attrs.GetAttr<bool>("inplace"));
  if (mode == "row" || mode == "triangular_start" || mode == "triangular_end") {
    CHECK_EQ_OR_RETURN(inputs.size(), 3);
    ctx->bias_requires_grad = inputs.at(2)->requires_grad();
  } else {
    CHECK_EQ_OR_RETURN(inputs.size(), 2);
    ctx->input_size = 2;
  }
  ctx->mode = mode;
  ctx->inplace = inplace;
  ctx->input_requires_grad = inputs.at(0)->requires_grad();
  if (!ctx->input_requires_grad) { return Maybe<void>::Ok(); }

  ctx->scale = JUST(composed_attrs.GetAttr<float>("scale"));

  ctx->SaveTensorForBackward(outputs.at(0));
  return Maybe<void>::Ok();
}

Maybe<void> FusedMSASoftmax::Apply(const FusedMSASoftmaxCaptureState* ctx,
                                   const TensorTuple& out_grads, TensorTuple* in_grads) const {
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // dy
  if (!ctx->input_requires_grad && !ctx->bias_requires_grad) { return Maybe<void>::Ok(); }
  in_grads->resize(ctx->input_size);

  const std::shared_ptr<oneflow::one::Tensor>& y = ctx->SavedTensors().at(0);
  const std::shared_ptr<oneflow::one::Tensor>& input_grad = JUST(
      functional::FusedMSASoftmaxGrad(y, out_grads.at(0), ctx->scale, ctx->mode, ctx->inplace));

  in_grads->at(0) = input_grad;
  if (ctx->bias_requires_grad) {
    in_grads->at(2) = JUST(
        functional::ScalarMul(1 / ctx->scale, JUST(functional::ReduceSum(input_grad, {0}, true))));
  }

  return Maybe<void>::Ok();
}

struct FusedMSABiasaddSigmoidMulCaptureState : public AutoGradCaptureState {
  bool input_requires_grad = false;
};

class FusedMSABiasaddSigmoidMul : public OpExprGradFunction<FusedMSABiasaddSigmoidMulCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const UserOpExpr* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);
    base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
    return Maybe<void>::Ok();
  }
  Maybe<void> Capture(FusedMSABiasaddSigmoidMulCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    bool inplace = JUST(composed_attrs.GetAttr<bool>("inplace"));
    CHECK_EQ_OR_RETURN(inplace, false);
    ctx->input_requires_grad = inputs.at(1)->requires_grad();
    ctx->SaveTensorForBackward(inputs.at(0));
    ctx->SaveTensorForBackward(inputs.at(1));
    ctx->SaveTensorForBackward(inputs.at(2));
    return Maybe<void>::Ok();
  }
  Maybe<void> Apply(const FusedMSABiasaddSigmoidMulCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    if (!ctx->input_requires_grad) return Maybe<void>::Ok();
    in_grads->resize(3);
    const std::shared_ptr<oneflow::one::Tensor>& b = ctx->SavedTensors().at(0);
    const std::shared_ptr<oneflow::one::Tensor>& g = ctx->SavedTensors().at(1);
    const std::shared_ptr<oneflow::one::Tensor>& x = ctx->SavedTensors().at(2);
    const std::shared_ptr<oneflow::one::Tensor>& dg =
        JUST(functional::FusedMSABiasaddSigmoidMulGrad(out_grads.at(0), b, g, x, false));
    const auto n = dg->shape()->NumAxes();
    std::vector<int32_t> dims(n - 1);
    for (int i = 0; i < n - 1; ++i) dims.push_back(i);
    in_grads->at(0) = JUST(functional::ReduceSum(dg, dims, false));
    in_grads->at(1) = dg;
    in_grads->at(2) = out_grads.at(0);
    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

struct FusedMSABiasaddDropoutResidualCaptureState : public AutoGradCaptureState {
  bool bias_requires_grad = false;
  bool input_requires_grad = false;
  bool residual_requires_grad = false;
};

class FusedMSABiasaddDropoutResidual
    : public OpExprGradFunction<FusedMSABiasaddDropoutResidualCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const UserOpExpr* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);
    base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
    return Maybe<void>::Ok();
  }
  Maybe<void> Capture(FusedMSABiasaddDropoutResidualCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    bool inplace = JUST(composed_attrs.GetAttr<bool>("inplace"));
    CHECK_EQ_OR_RETURN(inplace, false);
    ctx->bias_requires_grad = inputs.at(0)->requires_grad();
    ctx->input_requires_grad = inputs.at(1)->requires_grad();
    ctx->residual_requires_grad = inputs.at(3)->requires_grad();
    ctx->SaveTensorForBackward(inputs.at(2));
    return Maybe<void>::Ok();
  }
  Maybe<void> Apply(const FusedMSABiasaddDropoutResidualCaptureState* ctx,
                    const TensorTuple& out_grads, TensorTuple* in_grads) const override {
    if (!ctx->input_requires_grad) return Maybe<void>::Ok();
    in_grads->resize(4);
    const std::shared_ptr<oneflow::one::Tensor>& mask = ctx->SavedTensors().at(0);
    const std::shared_ptr<oneflow::one::Tensor>& dx =
        JUST(functional::FusedMSABiasaddDropoutResidualGrad(out_grads.at(0), mask, false));
    in_grads->at(1) = dx;
    auto n = dx->ndim();
    std::vector<int32_t> dims(n - 1);
    for (int i = 0; i < n - 1; i++) dims[i] = i;
    in_grads->at(0) = JUST(functional::ReduceSum(dx, dims, false));
    in_grads->at(3) = out_grads.at(0);
    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

struct FusedMSATmuState : public AutoGradCaptureState {
  bool input_requires_grad = false;
};

class FusedMSATmu : public OpExprGradFunction<FusedMSATmuState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const UserOpExpr* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);
    base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
    return Maybe<void>::Ok();
  }
  Maybe<void> Capture(FusedMSATmuState* ctx, const TensorTuple& inputs, const TensorTuple& outputs,
                      const AttrMap& attrs) const override {
    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    bool inplace = JUST(composed_attrs.GetAttr<bool>("inplace"));
    CHECK_EQ_OR_RETURN(inplace, false);
    ctx->input_requires_grad = inputs.at(0)->requires_grad();
    ctx->SaveTensorForBackward(inputs.at(0));
    ctx->SaveTensorForBackward(inputs.at(1));
    ctx->SaveTensorForBackward(inputs.at(2));
    ctx->SaveTensorForBackward(inputs.at(3));
    ctx->SaveTensorForBackward(inputs.at(4));
    return Maybe<void>::Ok();
  }
  Maybe<void> Apply(const FusedMSATmuState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    if (!ctx->input_requires_grad) return Maybe<void>::Ok();
    in_grads->resize(6);
    const std::shared_ptr<oneflow::one::Tensor>& x1 = ctx->SavedTensors().at(0);
    const std::shared_ptr<oneflow::one::Tensor>& b1 = ctx->SavedTensors().at(1);
    const std::shared_ptr<oneflow::one::Tensor>& x2 = ctx->SavedTensors().at(2);
    const std::shared_ptr<oneflow::one::Tensor>& b2 = ctx->SavedTensors().at(3);
    const std::shared_ptr<oneflow::one::Tensor>& mask = ctx->SavedTensors().at(4);
    const std::shared_ptr<oneflow::one::TensorTuple>& dx12 =
        JUST(functional::FusedMSATmuGrad(out_grads.at(0), x1, b1, x2, b2, mask, false));
    in_grads->at(0) = dx12->at(0);
    in_grads->at(2) = dx12->at(1);
    auto n = dx12->at(0)->ndim();
    std::vector<int32_t> dims(n - 1);
    for (int i = 0; i < n - 1; i++) dims[i] = i;
    in_grads->at(1) = JUST(functional::ReduceSum(dx12->at(0), dims, false));
    in_grads->at(3) = JUST(functional::ReduceSum(dx12->at(1), dims, false));
    in_grads->at(5) = out_grads.at(0);
    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("fused_msa_softmax", FusedMSASoftmax);
REGISTER_OP_EXPR_GRAD_FUNCTION("fused_msa_biasadd_sigmoid_mul", FusedMSABiasaddSigmoidMul);
REGISTER_OP_EXPR_GRAD_FUNCTION("fused_msa_biasadd_dropout_residual",
                               FusedMSABiasaddDropoutResidual);
REGISTER_OP_EXPR_GRAD_FUNCTION("fused_msa_tmu", FusedMSATmu);
}  // namespace one
}  // namespace oneflow
