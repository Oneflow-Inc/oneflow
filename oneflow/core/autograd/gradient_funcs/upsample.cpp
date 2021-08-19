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
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"

namespace oneflow {
namespace one {

struct UpsampleCaptureState : public AutoGradCaptureState {
  bool requires_grad;
  float height_scale;
  float width_scale;
  float align_corners;
  std::string data_format;
  std::string interpolation;
};

class Upsample : public OpExprGradFunction<UpsampleCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(UpsampleCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const UpsampleCaptureState* ctx, const TensorTuple& out_grads,
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

Maybe<void> Upsample::Capture(UpsampleCaptureState* ctx, const TensorTuple& inputs,
                              const TensorTuple& outputs, const AttrMap& attrs) const {
  ctx->requires_grad = inputs.at(0)->requires_grad();
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->height_scale = JUST(composed_attrs.GetAttr<float>("height_scale"));
  ctx->width_scale = JUST(composed_attrs.GetAttr<float>("width_scale"));
  ctx->align_corners = JUST(composed_attrs.GetAttr<bool>("align_corners"));
  ctx->data_format = JUST(composed_attrs.GetAttr<std::string>("data_format"));
  ctx->interpolation = JUST(composed_attrs.GetAttr<std::string>("interpolation"));
  ctx->SaveTensorForBackward(inputs.at(0));
  return Maybe<void>::Ok();
}

Maybe<void> Upsample::Apply(const UpsampleCaptureState* ctx, const TensorTuple& out_grads,
                            TensorTuple* in_grads) const {
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);

  MutableAttrMap attrs;
  JUST(attrs.SetAttr<float>("height_scale", ctx->height_scale));
  JUST(attrs.SetAttr<float>("width_scale", ctx->width_scale));
  JUST(attrs.SetAttr<bool>("align_corners", ctx->align_corners));
  JUST(attrs.SetAttr<std::string>("data_format", ctx->data_format));
  JUST(attrs.SetAttr<std::string>("interpolation", ctx->interpolation));
  const std::shared_ptr<oneflow::one::Tensor>& x = ctx->SavedTensors().at(0);
  in_grads->resize(1);
  in_grads->at(0) = JUST(OpInterpUtil::Dispatch<Tensor>(*grad_op_, {out_grads.at(0), x}, attrs));
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("upsample", Upsample);

struct UpsampleNearest2DCaptureState : public AutoGradCaptureState {
  bool requires_grad;
  float height_scale;
  float width_scale;
  std::string data_format;
};

class UpsampleNearest2D : public OpExprGradFunction<UpsampleNearest2DCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override { return Maybe<void>::Ok(); }

  Maybe<void> Capture(UpsampleNearest2DCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);
    CHECK_EQ_OR_RETURN(outputs.size(), 1);
    ctx->requires_grad = inputs.at(0)->requires_grad();
    if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    ctx->height_scale = JUST(composed_attrs.GetAttr<float>("height_scale"));
    ctx->width_scale = JUST(composed_attrs.GetAttr<float>("width_scale"));
    ctx->data_format = JUST(composed_attrs.GetAttr<std::string>("data_format"));
    ctx->SaveTensorForBackward(inputs.at(0));
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const UpsampleNearest2DCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    MutableAttrMap attrs;
    const std::shared_ptr<oneflow::one::Tensor>& x = ctx->SavedTensors().at(0);
    in_grads->resize(1);
    in_grads->at(0) = JUST(functional::UpsampleNearest2DGrad(out_grads.at(0), x, ctx->height_scale,
                                                             ctx->width_scale, ctx->data_format));

    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("upsample_nearest_2d", UpsampleNearest2D);

struct UpsampleBilinear2DCaptureState : public AutoGradCaptureState {
  bool requires_grad;
  float height_scale;
  float width_scale;
  bool align_corners;
  std::string data_format;
};

class UpsampleBilinear2D : public OpExprGradFunction<UpsampleBilinear2DCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override { return Maybe<void>::Ok(); }

  Maybe<void> Capture(UpsampleBilinear2DCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);
    CHECK_EQ_OR_RETURN(outputs.size(), 1);
    ctx->requires_grad = inputs.at(0)->requires_grad();
    if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    ctx->height_scale = JUST(composed_attrs.GetAttr<float>("height_scale"));
    ctx->width_scale = JUST(composed_attrs.GetAttr<float>("width_scale"));
    ctx->align_corners = JUST(composed_attrs.GetAttr<bool>("align_corners"));
    ctx->data_format = JUST(composed_attrs.GetAttr<std::string>("data_format"));
    ctx->SaveTensorForBackward(inputs.at(0));
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const UpsampleBilinear2DCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    MutableAttrMap attrs;
    const std::shared_ptr<oneflow::one::Tensor>& x = ctx->SavedTensors().at(0);
    in_grads->resize(1);
    in_grads->at(0) = JUST(functional::UpsampleBilinear2DGrad(out_grads.at(0), x, ctx->height_scale,
                                                              ctx->width_scale, ctx->align_corners,
                                                              ctx->data_format));

    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("upsample_bilinear_2d", UpsampleBilinear2D);

struct UpsampleLinear1DCaptureState : public AutoGradCaptureState {
  bool requires_grad;
  float scale_factor;
  bool align_corners;
  std::string data_format;
};

class UpsampleLinear1D : public OpExprGradFunction<UpsampleLinear1DCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override { return Maybe<void>::Ok(); }

  Maybe<void> Capture(UpsampleLinear1DCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);
    CHECK_EQ_OR_RETURN(outputs.size(), 1);
    ctx->requires_grad = inputs.at(0)->requires_grad();
    if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    ctx->scale_factor = JUST(composed_attrs.GetAttr<float>("scale_factor"));
    ctx->align_corners = JUST(composed_attrs.GetAttr<bool>("align_corners"));
    ctx->data_format = JUST(composed_attrs.GetAttr<std::string>("data_format"));
    ctx->SaveTensorForBackward(inputs.at(0));
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const UpsampleLinear1DCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    MutableAttrMap attrs;
    const std::shared_ptr<oneflow::one::Tensor>& x = ctx->SavedTensors().at(0);
    in_grads->resize(1);
    in_grads->at(0) = JUST(functional::UpsampleLinear1DGrad(out_grads.at(0), x, ctx->scale_factor,
                                                            ctx->align_corners, ctx->data_format));

    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("upsample_linear_1d", UpsampleLinear1D);

struct UpsampleNearest1DCaptureState : public AutoGradCaptureState {
  bool requires_grad;
  float scale_factor;
  std::string data_format;
};

class UpsampleNearest1D : public OpExprGradFunction<UpsampleNearest1DCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override { return Maybe<void>::Ok(); }

  Maybe<void> Capture(UpsampleNearest1DCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);
    CHECK_EQ_OR_RETURN(outputs.size(), 1);
    ctx->requires_grad = inputs.at(0)->requires_grad();
    if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    ctx->scale_factor = JUST(composed_attrs.GetAttr<float>("scale_factor"));
    ctx->data_format = JUST(composed_attrs.GetAttr<std::string>("data_format"));
    ctx->SaveTensorForBackward(inputs.at(0));
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const UpsampleNearest1DCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    MutableAttrMap attrs;
    const std::shared_ptr<oneflow::one::Tensor>& x = ctx->SavedTensors().at(0);
    in_grads->resize(1);
    in_grads->at(0) = JUST(
        functional::UpsampleNearest1DGrad(out_grads.at(0), x, ctx->scale_factor, ctx->data_format));

    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("upsample_nearest_1d", UpsampleNearest1D);

struct UpsampleBicubic2DCaptureState : public AutoGradCaptureState {
  bool requires_grad;
  float height_scale;
  float width_scale;
  bool align_corners;
  std::string data_format;
};

class UpsampleBicubic2D : public OpExprGradFunction<UpsampleBicubic2DCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override { return Maybe<void>::Ok(); }

  Maybe<void> Capture(UpsampleBicubic2DCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);
    CHECK_EQ_OR_RETURN(outputs.size(), 1);
    ctx->requires_grad = inputs.at(0)->requires_grad();
    if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    ctx->height_scale = JUST(composed_attrs.GetAttr<float>("height_scale"));
    ctx->width_scale = JUST(composed_attrs.GetAttr<float>("width_scale"));
    ctx->align_corners = JUST(composed_attrs.GetAttr<bool>("align_corners"));
    ctx->data_format = JUST(composed_attrs.GetAttr<std::string>("data_format"));
    ctx->SaveTensorForBackward(inputs.at(0));
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const UpsampleBicubic2DCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    MutableAttrMap attrs;
    const std::shared_ptr<oneflow::one::Tensor>& x = ctx->SavedTensors().at(0);
    in_grads->resize(1);
    in_grads->at(0) = JUST(functional::UpsampleBicubic2DGrad(out_grads.at(0), x, ctx->height_scale,
                                                             ctx->width_scale, ctx->align_corners,
                                                             ctx->data_format));
    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("upsample_bicubic_2d", UpsampleBicubic2D);

struct UpsampleNearest3DCaptureState : public AutoGradCaptureState {
  bool requires_grad;
  float depth_scale;
  float height_scale;
  float width_scale;
  std::string data_format;
};

class UpsampleNearest3D : public OpExprGradFunction<UpsampleNearest3DCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override { return Maybe<void>::Ok(); }

  Maybe<void> Capture(UpsampleNearest3DCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);
    CHECK_EQ_OR_RETURN(outputs.size(), 1);
    ctx->requires_grad = inputs.at(0)->requires_grad();
    if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    ctx->depth_scale = JUST(composed_attrs.GetAttr<float>("depth_scale"));
    ctx->height_scale = JUST(composed_attrs.GetAttr<float>("height_scale"));
    ctx->width_scale = JUST(composed_attrs.GetAttr<float>("width_scale"));
    ctx->data_format = JUST(composed_attrs.GetAttr<std::string>("data_format"));
    ctx->SaveTensorForBackward(inputs.at(0));
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const UpsampleNearest3DCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    MutableAttrMap attrs;
    const std::shared_ptr<oneflow::one::Tensor>& x = ctx->SavedTensors().at(0);
    in_grads->resize(1);
    in_grads->at(0) = JUST(functional::UpsampleNearest3DGrad(out_grads.at(0), x, ctx->depth_scale,
                                                             ctx->height_scale, ctx->width_scale,
                                                             ctx->data_format));

    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("upsample_nearest_3d", UpsampleNearest3D);

struct UpsampleTrilinear3DCaptureState : public AutoGradCaptureState {
  bool requires_grad;
  float depth_scale;
  float height_scale;
  float width_scale;
  bool align_corners;
  std::string data_format;
};

class UpsampleTrilinear3D : public OpExprGradFunction<UpsampleTrilinear3DCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override { return Maybe<void>::Ok(); }

  Maybe<void> Capture(UpsampleTrilinear3DCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);
    CHECK_EQ_OR_RETURN(outputs.size(), 1);
    ctx->requires_grad = inputs.at(0)->requires_grad();
    if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    ctx->depth_scale = JUST(composed_attrs.GetAttr<float>("depth_scale"));
    ctx->height_scale = JUST(composed_attrs.GetAttr<float>("height_scale"));
    ctx->width_scale = JUST(composed_attrs.GetAttr<float>("width_scale"));
    ctx->align_corners = JUST(composed_attrs.GetAttr<bool>("align_corners"));
    ctx->data_format = JUST(composed_attrs.GetAttr<std::string>("data_format"));
    ctx->SaveTensorForBackward(inputs.at(0));
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const UpsampleTrilinear3DCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    MutableAttrMap attrs;
    const std::shared_ptr<oneflow::one::Tensor>& x = ctx->SavedTensors().at(0);
    in_grads->resize(1);
    in_grads->at(0) = JUST(functional::UpsampleTrilinear3DGrad(
        out_grads.at(0), x, ctx->depth_scale, ctx->height_scale, ctx->width_scale,
        ctx->align_corners, ctx->data_format));

    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("upsample_trilinear_3d", UpsampleTrilinear3D);

}  // namespace one
}  // namespace oneflow
