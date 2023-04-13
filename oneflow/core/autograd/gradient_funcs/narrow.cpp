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
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/job/lazy_mode.h"
#include "oneflow/core/framework/nd_sbp.h"

namespace oneflow {
namespace one {

struct NarrowCaptureState : public AutoGradCaptureState {
  bool requires_grad;
  Shape shape;
  int64_t dim;
  int64_t start;
  int64_t length;
};

class Narrow : public OpExprGradFunction<NarrowCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
    base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(NarrowCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);   // NOLINT(maybe-need-error-msg)
    CHECK_EQ_OR_RETURN(outputs.size(), 1);  // NOLINT(maybe-need-error-msg)
    ctx->requires_grad = inputs.at(0)->requires_grad();
    if (!ctx->requires_grad) { return Maybe<void>::Ok(); }

    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    ctx->dim = JUST(composed_attrs.GetAttr<int64_t>("dim"));
    ctx->start = JUST(composed_attrs.GetAttr<int64_t>("start"));
    ctx->length = JUST(composed_attrs.GetAttr<int64_t>("length"));
    if (LazyMode::is_enabled()) {
      ctx->SaveTensorForBackward(inputs.at(0));
    } else {
      ctx->shape = *(inputs.at(0)->shape());
    }
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const NarrowCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    const auto& dy = out_grads.at(0);
    if (ctx->requires_grad) {
      std::shared_ptr<Tensor> like;
      if (LazyMode::is_enabled()) {
        like = ctx->SavedTensors().at(0);
      } else if (dy->is_local()) {
        like = JUST(functional::Empty(ctx->shape, dy->dtype(), JUST(dy->device()),
                                      ctx->requires_grad, /*pin_memory=*/false));
      } else {
        like = JUST(
            functional::GlobalEmpty(ctx->shape, dy->dtype(), JUST(dy->parallel_desc()),
                                    *JUST(private_details::RawGetSbpList(JUST(dy->nd_sbp())))));
      }
      in_grads->resize(1);
      in_grads->at(0) = JUST(functional::NarrowGrad(dy, like, ctx->dim, ctx->start, ctx->length));
    }
    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("narrow", Narrow);

}  // namespace one
}  // namespace oneflow
