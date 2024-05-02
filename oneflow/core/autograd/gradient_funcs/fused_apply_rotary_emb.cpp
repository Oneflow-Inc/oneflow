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

namespace oneflow {
namespace one {

struct FusedApplyRotaryEmbCaptureState : public AutoGradCaptureState {
  bool requires_grad;
  std::string x_layout{};
  std::string output_layout{};
  std::string mode{};
  int64_t tensor_index{};
  int64_t k_size{};
  float base;
  int64_t rotary_size{};
};

class FusedApplyRotaryEmb : public OpExprGradFunction<FusedApplyRotaryEmbCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(FusedApplyRotaryEmbCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const FusedApplyRotaryEmbCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
};

Maybe<void> FusedApplyRotaryEmb::Init(const OpExpr& op) {
  const UserOpExpr* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  return Maybe<void>::Ok();
}

Maybe<void> FusedApplyRotaryEmb::Capture(FusedApplyRotaryEmbCaptureState* ctx,
                                         const TensorTuple& inputs, const TensorTuple& outputs,
                                         const AttrMap& attrs) const {
  CHECK_OR_RETURN((inputs.size() >= 1) && (inputs.size() <= 4))
      << Error::RuntimeError() << "the inputs size of fusedapplyrotaryembgrad\
    should between 1 and 4";
  ctx->requires_grad = inputs.at(0)->requires_grad();
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }

  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->SaveTensorForBackward(inputs.at(0));
  if (inputs.size() == 2)  // position_ids
    ctx->SaveTensorForBackward(inputs.at(1));

  if (inputs.size() == 3) {  // cos, sin
    ctx->SaveTensorForBackward(inputs.at(1));
    ctx->SaveTensorForBackward(inputs.at(2));
  }

  if (inputs.size() == 4) {  // cos, sin, position_ids;
    ctx->SaveTensorForBackward(inputs.at(1));
    ctx->SaveTensorForBackward(inputs.at(2));
    ctx->SaveTensorForBackward(inputs.at(3));
  }

  ctx->x_layout = JUST(composed_attrs.GetAttr<std::string>("x_layout"));
  ctx->output_layout = JUST(composed_attrs.GetAttr<std::string>("output_layout"));
  ctx->mode = JUST(composed_attrs.GetAttr<std::string>("mode"));
  ctx->tensor_index = JUST(composed_attrs.GetAttr<int64_t>("tensor_index"));
  ctx->k_size = JUST(composed_attrs.GetAttr<int64_t>("k_size"));
  ctx->base = JUST(composed_attrs.GetAttr<float>("base"));
  ctx->rotary_size = JUST(composed_attrs.GetAttr<int64_t>("rotary_size"));

  return Maybe<void>::Ok();
}

Maybe<void> FusedApplyRotaryEmb::Apply(const FusedApplyRotaryEmbCaptureState* ctx,
                                       const TensorTuple& out_grads, TensorTuple* in_grads) const {
  CHECK_EQ_OR_RETURN(out_grads.size(), 1)
      << Error::RuntimeError() << "fusedapplyrotaryembgrad outgrad size should be 1";
  const auto& saved_tensors = ctx->SavedTensors();

  CHECK_OR_RETURN((saved_tensors.size() >= 1) && (saved_tensors.size() <= 4))
      << Error::RuntimeError() << "the saved_tensors of fusedapplyrotaryembgrad\
    should between 1 and 4";

  if (ctx->requires_grad) {
    if (saved_tensors.size() == 1) {  // x
      const auto& x = ctx->SavedTensors().at(0);
      in_grads->at(0) = JUST(functional::FusedApplyRotaryEmbGrad(
          x, out_grads.at(0), NullOpt /*cos*/, NullOpt /*sin*/, NullOpt /*position_ids*/,
          ctx->x_layout, ctx->output_layout, ctx->mode, ctx->tensor_index, ctx->k_size, ctx->base,
          ctx->rotary_size));
    }

    if (saved_tensors.size() == 2) {  // x, position_ids
      const auto& x = ctx->SavedTensors().at(0);
      const auto& position_ids = ctx->SavedTensors().at(1);
      in_grads->at(0) = JUST(functional::FusedApplyRotaryEmbGrad(
          x, out_grads.at(0), NullOpt /*cos*/, NullOpt /*sin*/, position_ids, ctx->x_layout,
          ctx->output_layout, ctx->mode, ctx->tensor_index, ctx->k_size, ctx->base,
          ctx->rotary_size));
    }

    if (saved_tensors.size() == 3) {  // x, cos, sin, position_ids
      const auto& x = ctx->SavedTensors().at(0);
      const auto& cos = ctx->SavedTensors().at(1);
      const auto& sin = ctx->SavedTensors().at(2);

      in_grads->at(0) = JUST(functional::FusedApplyRotaryEmbGrad(
          x, out_grads.at(0), cos, sin, NullOpt /*position_ids*/, ctx->x_layout, ctx->output_layout,
          ctx->mode, ctx->tensor_index, ctx->k_size, ctx->base, ctx->rotary_size));
    }

    if (saved_tensors.size() == 4) {
      const auto& x = ctx->SavedTensors().at(0);
      const auto& cos = ctx->SavedTensors().at(1);
      const auto& sin = ctx->SavedTensors().at(2);
      const auto& position_ids = ctx->SavedTensors().at(3);
      in_grads->at(0) = JUST(functional::FusedApplyRotaryEmbGrad(
          x, out_grads.at(0), cos, sin, position_ids, ctx->x_layout, ctx->output_layout, ctx->mode,
          ctx->tensor_index, ctx->k_size, ctx->base, ctx->rotary_size));
    }
  }

  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("fused_apply_rotary_emb", FusedApplyRotaryEmb);

}  // namespace one
}  // namespace oneflow
