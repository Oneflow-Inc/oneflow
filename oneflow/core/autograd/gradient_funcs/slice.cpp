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
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct SliceCaptureState : public AutoGradCaptureState {
  bool requires_grad;
  Shape like_shape;
  std::vector<int64_t> start;
  std::vector<int64_t> stop;
  std::vector<int64_t> step;
};

class Slice : public OpExprGradFunction<SliceCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);
    base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(SliceCaptureState* ctx, const TensorTuple& inputs, const TensorTuple& outputs,
                      const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);
    CHECK_EQ_OR_RETURN(outputs.size(), 1);
    ctx->requires_grad = inputs.at(0)->requires_grad();
    if (!ctx->requires_grad) { return Maybe<void>::Ok(); }

    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    ctx->start = JUST(composed_attrs.GetAttr<std::vector<int64_t>>("start"));
    ctx->stop = JUST(composed_attrs.GetAttr<std::vector<int64_t>>("stop"));
    ctx->step = JUST(composed_attrs.GetAttr<std::vector<int64_t>>("step"));
    ctx->like_shape = *(inputs.at(0)->shape());
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const SliceCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    in_grads->resize(1);
    in_grads->at(0) = JUST(
        functional::SliceGrad(out_grads.at(0), ctx->like_shape, ctx->start, ctx->stop, ctx->step));
    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

struct SliceUpdateCaptureState : public AutoGradCaptureState {
  bool requires_grad_x;
  bool requires_grad_update;
  std::vector<int64_t> start;
  std::vector<int64_t> stop;
  std::vector<int64_t> step;
};

class SliceUpdate : public OpExprGradFunction<SliceUpdateCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);

    base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(SliceUpdateCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 2);
    CHECK_EQ_OR_RETURN(outputs.size(), 1);
    ctx->requires_grad_x = inputs.at(0)->requires_grad();
    ctx->requires_grad_update = inputs.at(1)->requires_grad();
    if (!ctx->requires_grad_x && !ctx->requires_grad_update) { return Maybe<void>::Ok(); }

    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    ctx->start = JUST(composed_attrs.GetAttr<std::vector<int64_t>>("start"));
    ctx->stop = JUST(composed_attrs.GetAttr<std::vector<int64_t>>("stop"));
    ctx->step = JUST(composed_attrs.GetAttr<std::vector<int64_t>>("step"));

    if (ctx->requires_grad_x) { ctx->SaveTensorForBackward(inputs.at(1)); }
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const SliceUpdateCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    in_grads->resize(2);

    if (ctx->requires_grad_x) {
      const auto& update = ctx->SavedTensors().at(0);
      const auto& temp = JUST(functional::ZerosLike(update));
      in_grads->at(0) = JUST(functional::SliceUpdate(out_grads.at(0), temp, ctx->start, ctx->stop,
                                                     ctx->step, /*inplace=*/false));
    }
    if (ctx->requires_grad_update) {
      in_grads->at(1) = JUST(functional::Slice(out_grads.at(0), ctx->start, ctx->stop, ctx->step));
    }
    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("slice", Slice);
REGISTER_OP_EXPR_GRAD_FUNCTION("slice_update", SliceUpdate);

}  // namespace one
}  // namespace oneflow
