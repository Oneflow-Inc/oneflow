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
#include "oneflow/core/common/just.h"
#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/op_expr_grad_function.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/functional/functional_api.yaml.h"

namespace oneflow {
namespace one {

struct FillCaptureState : public AutoGradCaptureState {
  bool in_requires_grad = false;
  bool value_requires_grad = false;
};

class Fill : public OpExprGradFunction<FillCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(FillCaptureState* ctx, const TensorTuple& inputs, const TensorTuple& outputs,
                      const AttrMap& attrs) const override;
  Maybe<void> Apply(const FillCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
};

Maybe<void> Fill::Init(const OpExpr& op) {
  const UserOpExpr* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  return Maybe<void>::Ok();
}

Maybe<void> Fill::Capture(FillCaptureState* ctx, const TensorTuple& inputs,
                          const TensorTuple& outputs, const AttrMap& attrs) const {
  ctx->in_requires_grad = inputs[0]->requires_grad();
  return Maybe<void>::Ok();
}

Maybe<void> Fill::Apply(const FillCaptureState* ctx, const TensorTuple& out_grads,
                        TensorTuple* in_grads) const {
  CHECK_EQ_OR_RETURN(out_grads.size(), 1) << "out_grads.size() must be equal to 1.";
  in_grads->resize(1);
  if (ctx->in_requires_grad) { (*in_grads)[0] = JUST(functional::Fill(out_grads[0], 0)); }
  return Maybe<void>::Ok();
}

class FillTensor : public OpExprGradFunction<FillCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(FillCaptureState* ctx, const TensorTuple& inputs, const TensorTuple& outputs,
                      const AttrMap& attrs) const override;
  Maybe<void> Apply(const FillCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
};

Maybe<void> FillTensor::Init(const OpExpr& op) {
  const UserOpExpr* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  return Maybe<void>::Ok();
}

Maybe<void> FillTensor::Capture(FillCaptureState* ctx, const TensorTuple& inputs,
                                const TensorTuple& outputs, const AttrMap& attrs) const {
  ctx->in_requires_grad = inputs[0]->requires_grad();
  ctx->value_requires_grad = inputs[1]->requires_grad();
  return Maybe<void>::Ok();
}

Maybe<void> FillTensor::Apply(const FillCaptureState* ctx, const TensorTuple& out_grads,
                              TensorTuple* in_grads) const {
  CHECK_EQ_OR_RETURN(out_grads.size(), 1) << "out_grads.size() must be equal to 1.";
  in_grads->resize(2);
  if (ctx->value_requires_grad) {
    int32_t num_axes = out_grads[0]->shape()->NumAxes();
    std::vector<int32_t> axes_vec(num_axes);
    std::iota(axes_vec.begin(), axes_vec.end(), 0);
    (*in_grads)[1] = JUST(functional::ReduceSum(out_grads[0], axes_vec, /*keepdims=*/false));
  }
  if (ctx->in_requires_grad) { (*in_grads)[0] = JUST(functional::Fill(out_grads[0], 0)); }
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("fill_", Fill);
REGISTER_OP_EXPR_GRAD_FUNCTION("fill_tensor_", FillTensor);

}  // namespace one
}  // namespace oneflow
