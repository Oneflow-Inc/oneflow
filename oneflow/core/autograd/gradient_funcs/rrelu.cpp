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

struct RReLUCaptureState : public AutoGradCaptureState {
  bool requires_grad = true;
  float lower;
  float upper;
  bool training;
  int x_index = -1;
  int noise_data_index = -1;
};

class RReLU : public OpExprGradFunction<RReLUCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(RReLUCaptureState* ctx, const TensorTuple& inputs, const TensorTuple& outputs,
                      const AttrMap& attrs) const override;
  Maybe<void> Apply(const RReLUCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
};

Maybe<void> RReLU::Init(const OpExpr& op) {
  //   const UserOpExpr* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  //   CHECK_NOTNULL_OR_RETURN(fw_op_expr);
  //   base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  return Maybe<void>::Ok();
}

Maybe<void> RReLU::Capture(RReLUCaptureState* ctx, const TensorTuple& inputs,
                           const TensorTuple& outputs, const AttrMap& attrs) const {
  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->requires_grad = inputs.at(0)->requires_grad();
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
  ctx->lower = JUST(composed_attrs.GetAttr<float>("lower"));
  ctx->upper = JUST(composed_attrs.GetAttr<float>("upper"));
  ctx->training = JUST(composed_attrs.GetAttr<bool>("training"));
  ctx->x_index = ctx->SaveTensorForBackward(inputs[0]);
  ctx->noise_data_index = ctx->SaveTensorForBackward(outputs[1]);  // output noise data
  return Maybe<void>::Ok();
}

Maybe<void> RReLU::Apply(const RReLUCaptureState* ctx, const TensorTuple& out_grads,
                         TensorTuple* in_grads) const {
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
  const auto& saved_tensors = ctx->SavedTensors();
  if (!ctx->training) {
    float scale = (ctx->lower + ctx->upper) / 2;
    const auto& x = saved_tensors.at(ctx->x_index);
    in_grads->at(0) = JUST(functional::LeakyReluGrad(x, out_grads.at(0), scale));
    return Maybe<void>::Ok();

  } else {
    const auto& noise_data = saved_tensors.at(ctx->noise_data_index);
    in_grads->at(0) =
          JUST(functional::RReluGrad(out_grads.at(0), noise_data ));
          return Maybe<void>::Ok();
  }
}

REGISTER_OP_EXPR_GRAD_FUNCTION("rrelu", RReLU);

}  // namespace one
}  // namespace oneflow
