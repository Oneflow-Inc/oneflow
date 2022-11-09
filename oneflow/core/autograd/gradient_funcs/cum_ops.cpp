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

namespace oneflow {
namespace one {

struct CumCaptureState : public AutoGradCaptureState {
  bool requires_grad = false;
  int32_t dim = 0;
};

template<typename StateT>
class CumGrad : public OpExprGradFunction<StateT> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
    base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
    return Maybe<void>::Ok();
  }

 protected:
  AttrMap base_attrs_;
};

class CumsumGrad : public CumGrad<CumCaptureState> {
 public:
  Maybe<void> Capture(CumCaptureState* ctx, const TensorTuple& inputs, const TensorTuple& outputs,
                      const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);  // NOLINT(maybe-need-error-msg)
    ctx->requires_grad = inputs.at(0)->requires_grad();
    if (!ctx->requires_grad) { return Maybe<void>::Ok(); }

    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    ctx->dim = JUST(composed_attrs.GetAttr<int64_t>("dim"));
    return Maybe<void>::Ok();
  }
  Maybe<void> Apply(const CumCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)
    in_grads->resize(1);
    if (ctx->requires_grad) {
      std::vector<int32_t> flip_dim(1, ctx->dim);
      (*in_grads)[0] = JUST(
          functional::Flip(JUST(functional::Cumsum(JUST(functional::Flip(out_grads[0], flip_dim)),
                                                   ctx->dim, out_grads[0]->dtype())),
                           flip_dim));
    }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("cumsum", CumsumGrad);

class CumProdGrad : public CumGrad<CumCaptureState> {
 public:
  Maybe<void> Capture(CumCaptureState* ctx, const TensorTuple& inputs, const TensorTuple& outputs,
                      const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);  // NOLINT(maybe-need-error-msg)
    ctx->requires_grad = inputs.at(0)->requires_grad();
    if (!ctx->requires_grad) { return Maybe<void>::Ok(); }

    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    ctx->dim = JUST(composed_attrs.GetAttr<int64_t>("dim"));
    ctx->SaveTensorForBackward(outputs.at(0));
    ctx->SaveTensorForBackward(inputs.at(0));
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const CumCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)
    in_grads->resize(1);
    if (ctx->requires_grad) {
      in_grads->at(0) = JUST(functional::CumprodGrad(out_grads.at(0), ctx->SavedTensors().at(0),
                                                     ctx->SavedTensors().at(1), ctx->dim));
    }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("cumprod", CumProdGrad);

}  // namespace one
}  // namespace oneflow
