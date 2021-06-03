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
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_expr_helper.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"

namespace oneflow {
namespace one {

struct UpsampleInterpState : public OpExprInterpState {
  bool requires_grad;
  float height_scale;
  float width_scale;
  float align_corners;
  std::string data_format;
  std::string interpolation;
};

class Upsample : public OpExprGradFunction<UpsampleInterpState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(UpsampleInterpState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const UpsampleInterpState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
  std::shared_ptr<OpExpr> grad_op_;
};

Maybe<void> Upsample::Init(const OpExpr& op) {
  const UserOpExpr* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  const std::string& op_name = fw_op_expr->op_name();
  const float height_scale = 1.0;
  const float width_scale = 1.0;
  const bool align_corners = false;
  const std::string data_format = "NCHW";
  const std::string interpolation = "nearest";
  grad_op_ =
      JUST(op_expr_helper::UpsampleGradOp(height_scale, width_scale, align_corners, data_format,
                                          interpolation, GradientOpName(op_name)));
  return Maybe<void>::Ok();
}

Maybe<void> Upsample::Capture(UpsampleInterpState* ctx, const TensorTuple& inputs,
                              const TensorTuple& outputs, const AttrMap& attrs) const {
  ctx->requires_grad = inputs.at(0)->requires_grad();
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->height_scale = JUST(composed_attrs.GetAttr<float>("height_scale"));
  ctx->width_scale = JUST(composed_attrs.GetAttr<float>("width_scale"));
  ctx->align_corners = JUST(composed_attrs.GetAttr<bool>("align_corners"));
  ctx->data_format = JUST(composed_attrs.GetAttr<std::string>("data_format"));
  ctx->interpolation = JUST(composed_attrs.GetAttr<std::string>("interpolation"));
  return Maybe<void>::Ok();
}

Maybe<void> Upsample::Apply(const UpsampleInterpState* ctx, const TensorTuple& out_grads,
                            TensorTuple* in_grads) const {
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);

  MutableAttrMap attrs;
  JUST(attrs.SetAttr<float>("height_scale", ctx->height_scale));
  JUST(attrs.SetAttr<float>("width_scale", ctx->width_scale));
  JUST(attrs.SetAttr<bool>("align_corners", ctx->align_corners));
  JUST(attrs.SetAttr<std::string>("data_format", ctx->data_format));
  JUST(attrs.SetAttr<std::string>("interpolation", ctx->interpolation));
  in_grads->resize(1);
  in_grads->at(0) = JUST(OpInterpUtil::Dispatch<Tensor>(*grad_op_, {out_grads.at(0)}, attrs));
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("upsample", Upsample);

}  // namespace one
}  // namespace oneflow
