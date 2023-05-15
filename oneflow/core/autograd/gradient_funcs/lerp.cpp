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
#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/op_expr_grad_function.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

const int32_t INPUT_LEN = 3;
struct LerpCaptureState : public AutoGradCaptureState {
  std::vector<bool> requires_grad;
};
struct ScalarLerpCaptureState : public AutoGradCaptureState {
  std::vector<bool> requires_grad;
  Scalar operand;
};

class LerpGrad : public OpExprGradFunction<LerpCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override { return Maybe<void>::Ok(); }

  Maybe<void> Capture(LerpCaptureState* ctx, const TensorTuple& inputs, const TensorTuple& outputs,
                      const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), INPUT_LEN);
    CHECK_EQ_OR_RETURN(outputs.size(), 1);

    for (int i = 0; i < INPUT_LEN; i++) {
      ctx->requires_grad.push_back(inputs.at(i)->requires_grad());
      ctx->SaveTensorForBackward(inputs.at(i));
    }

    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const LerpCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    const auto& out_diff = out_grads.at(0);

    const auto& start = ctx->SavedTensors().at(0);
    const auto& end = ctx->SavedTensors().at(1);
    const auto& weight = ctx->SavedTensors().at(2);

    auto result = JUST(functional::LerpGrad(start, end, weight, out_diff));
    CHECK_EQ_OR_RETURN(result->size(), INPUT_LEN);

    in_grads->resize(INPUT_LEN);
    for (int i = 0; i < INPUT_LEN; i++) {
      if (ctx->requires_grad[i]) { in_grads->at(i) = result->at(i); }
    }
    return Maybe<void>::Ok();
  }
};

class ScalarLerpGrad : public OpExprGradFunction<ScalarLerpCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(ScalarLerpCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), INPUT_LEN - 1);
    CHECK_EQ_OR_RETURN(outputs.size(), 1);

    for (int i = 0; i < INPUT_LEN - 1; i++) {
      ctx->requires_grad.push_back(inputs.at(i)->requires_grad());
      ctx->SaveTensorForBackward(inputs.at(i));
    }

    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    bool has_float_operand = JUST(composed_attrs.GetAttr<bool>("has_float_operand"));
    if (has_float_operand) {
      ctx->operand = Scalar(JUST(composed_attrs.GetAttr<double>("float_operand")));
    } else {
      ctx->operand = Scalar(JUST(composed_attrs.GetAttr<int64_t>("int_operand")));
    }

    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const ScalarLerpCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    const auto& out_diff = out_grads.at(0);

    const auto& start = ctx->SavedTensors().at(0);
    const auto& end = ctx->SavedTensors().at(1);

    auto result = JUST(functional::ScalarLerpGrad(start, end, out_diff, ctx->operand));
    CHECK_EQ_OR_RETURN(result->size(), INPUT_LEN - 1);

    in_grads->resize(INPUT_LEN - 1);
    for (int i = 0; i < INPUT_LEN - 1; i++) {
      if (ctx->requires_grad[i]) { in_grads->at(i) = result->at(i); }
    }
    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("lerp", LerpGrad);
REGISTER_OP_EXPR_GRAD_FUNCTION("scalar_lerp", ScalarLerpGrad);

}  // namespace one
}  // namespace oneflow
