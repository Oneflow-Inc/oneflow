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

struct SliceGradGradCaptureState : public AutoGradCaptureState {
  std::vector<int64_t> start;
  std::vector<int64_t> stop;
  std::vector<int64_t> step;
};

class SliceGradGrad : public OpExprGradFunction<SliceGradGradCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
    base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(SliceGradGradCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);   // NOLINT(maybe-need-error-msg)
    CHECK_EQ_OR_RETURN(outputs.size(), 1);  // NOLINT(maybe-need-error-msg)
    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    ctx->start = JUST(composed_attrs.GetAttr<std::vector<int64_t>>("start"));
    ctx->stop = JUST(composed_attrs.GetAttr<std::vector<int64_t>>("stop"));
    ctx->step = JUST(composed_attrs.GetAttr<std::vector<int64_t>>("step"));
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const SliceGradGradCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    in_grads->resize(1);
    in_grads->at(0) = JUST(functional::Slice(out_grads.at(0), ctx->start, ctx->stop, ctx->step,
                                             /*enable_view_slice=*/false));
    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("slice_grad", SliceGradGrad);

}  // namespace one
}  // namespace oneflow
