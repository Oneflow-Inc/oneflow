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
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/common/container_util.h"

namespace oneflow {
namespace one {

struct UpsampleCaptureState : public AutoGradCaptureState {
  bool requires_grad = false;
  double height_scale = 0.0;
  double width_scale = 0.0;
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
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  return Maybe<void>::Ok();
}

Maybe<void> Upsample::Capture(UpsampleCaptureState* ctx, const TensorTuple& inputs,
                              const TensorTuple& outputs, const AttrMap& attrs) const {
  ctx->requires_grad = inputs.at(0)->requires_grad();
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->height_scale = JUST(composed_attrs.GetAttr<double>("height_scale"));
  ctx->width_scale = JUST(composed_attrs.GetAttr<double>("width_scale"));
  ctx->align_corners = JUST(composed_attrs.GetAttr<bool>("align_corners"));
  ctx->data_format = JUST(composed_attrs.GetAttr<std::string>("data_format"));
  ctx->interpolation = JUST(composed_attrs.GetAttr<std::string>("interpolation"));
  ctx->SaveTensorForBackward(inputs.at(0));
  return Maybe<void>::Ok();
}

Maybe<void> Upsample::Apply(const UpsampleCaptureState* ctx, const TensorTuple& out_grads,
                            TensorTuple* in_grads) const {
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)

  const std::shared_ptr<oneflow::one::Tensor>& x = ctx->SavedTensors().at(0);
  in_grads->resize(1);
  JUST(oneflow::VectorAt(*in_grads, 0)) = JUST(functional::UpsampleGrad(
      JUST(oneflow::VectorAt(out_grads, 0)), x, ctx->height_scale, ctx->width_scale,
      ctx->align_corners, ctx->data_format, ctx->interpolation));
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("upsample", Upsample);

struct UpsampleNearest2DCaptureState : public AutoGradCaptureState {
  bool requires_grad = false;
  double height_scale = 0.0;
  double width_scale = 0.0;
  std::vector<int64_t> output_size;
  std::string data_format;
};

class UpsampleNearest2D : public OpExprGradFunction<UpsampleNearest2DCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override { return Maybe<void>::Ok(); }

  Maybe<void> Capture(UpsampleNearest2DCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);   // NOLINT(maybe-need-error-msg)
    CHECK_EQ_OR_RETURN(outputs.size(), 1);  // NOLINT(maybe-need-error-msg)
    ctx->requires_grad = inputs.at(0)->requires_grad();
    if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    ctx->height_scale = JUST(composed_attrs.GetAttr<double>("height_scale"));
    ctx->width_scale = JUST(composed_attrs.GetAttr<double>("width_scale"));
    if (composed_attrs.Has("output_size")) {
      ctx->output_size = JUST(composed_attrs.GetAttr<std::vector<int64_t>>("output_size"));
    }
    ctx->data_format = JUST(composed_attrs.GetAttr<std::string>("data_format"));
    ctx->SaveTensorForBackward(inputs.at(0));
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const UpsampleNearest2DCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)
    const std::shared_ptr<oneflow::one::Tensor>& x = ctx->SavedTensors().at(0);
    in_grads->resize(1);
    JUST(oneflow::VectorAt(*in_grads, 0)) = JUST(functional::UpsampleNearest2DGrad(
        JUST(oneflow::VectorAt(out_grads, 0)), x, ctx->height_scale, ctx->width_scale,
        ctx->output_size, ctx->data_format));

    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("upsample_nearest_2d", UpsampleNearest2D);

struct UpsampleBilinear2DCaptureState : public AutoGradCaptureState {
  bool requires_grad = false;
  double height_scale = 0.0;
  double width_scale = 0.0;
  bool align_corners;
  std::vector<int64_t> output_size;
  std::string data_format;
};

class UpsampleBilinear2D : public OpExprGradFunction<UpsampleBilinear2DCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override { return Maybe<void>::Ok(); }

  Maybe<void> Capture(UpsampleBilinear2DCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);   // NOLINT(maybe-need-error-msg)
    CHECK_EQ_OR_RETURN(outputs.size(), 1);  // NOLINT(maybe-need-error-msg)
    ctx->requires_grad = inputs.at(0)->requires_grad();
    if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    ctx->height_scale = JUST(composed_attrs.GetAttr<double>("height_scale"));
    ctx->width_scale = JUST(composed_attrs.GetAttr<double>("width_scale"));
    ctx->align_corners = JUST(composed_attrs.GetAttr<bool>("align_corners"));
    if (composed_attrs.Has("output_size")) {
      ctx->output_size = JUST(composed_attrs.GetAttr<std::vector<int64_t>>("output_size"));
    }
    ctx->data_format = JUST(composed_attrs.GetAttr<std::string>("data_format"));
    ctx->SaveTensorForBackward(inputs.at(0));
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const UpsampleBilinear2DCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)
    const std::shared_ptr<oneflow::one::Tensor>& x = ctx->SavedTensors().at(0);
    in_grads->resize(1);
    JUST(oneflow::VectorAt(*in_grads, 0)) = JUST(functional::UpsampleBilinear2DGrad(
        JUST(oneflow::VectorAt(out_grads, 0)), x, ctx->height_scale, ctx->width_scale,
        ctx->align_corners, ctx->output_size, ctx->data_format));

    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("upsample_bilinear_2d", UpsampleBilinear2D);

struct UpsampleLinear1DCaptureState : public AutoGradCaptureState {
  bool requires_grad = false;
  double scale_factor = 0.0;
  bool align_corners;
  std::vector<int64_t> output_size;
  std::string data_format;
};

class UpsampleLinear1D : public OpExprGradFunction<UpsampleLinear1DCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override { return Maybe<void>::Ok(); }

  Maybe<void> Capture(UpsampleLinear1DCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);   // NOLINT(maybe-need-error-msg)
    CHECK_EQ_OR_RETURN(outputs.size(), 1);  // NOLINT(maybe-need-error-msg)
    ctx->requires_grad = inputs.at(0)->requires_grad();
    if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    ctx->scale_factor = JUST(composed_attrs.GetAttr<double>("scale_factor"));
    ctx->align_corners = JUST(composed_attrs.GetAttr<bool>("align_corners"));
    if (composed_attrs.Has("output_size")) {
      ctx->output_size = JUST(composed_attrs.GetAttr<std::vector<int64_t>>("output_size"));
    }
    ctx->data_format = JUST(composed_attrs.GetAttr<std::string>("data_format"));
    ctx->SaveTensorForBackward(inputs.at(0));
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const UpsampleLinear1DCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)
    const std::shared_ptr<oneflow::one::Tensor>& x = ctx->SavedTensors().at(0);
    in_grads->resize(1);
    JUST(oneflow::VectorAt(*in_grads, 0)) = JUST(functional::UpsampleLinear1DGrad(
        JUST(oneflow::VectorAt(out_grads, 0)), x, ctx->scale_factor, ctx->align_corners,
        ctx->output_size, ctx->data_format));

    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("upsample_linear_1d", UpsampleLinear1D);

struct UpsampleNearest1DCaptureState : public AutoGradCaptureState {
  bool requires_grad = false;
  double scale_factor = 0.0;
  std::vector<int64_t> output_size;
  std::string data_format;
};

class UpsampleNearest1D : public OpExprGradFunction<UpsampleNearest1DCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override { return Maybe<void>::Ok(); }

  Maybe<void> Capture(UpsampleNearest1DCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);   // NOLINT(maybe-need-error-msg)
    CHECK_EQ_OR_RETURN(outputs.size(), 1);  // NOLINT(maybe-need-error-msg)
    ctx->requires_grad = inputs.at(0)->requires_grad();
    if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    ctx->scale_factor = JUST(composed_attrs.GetAttr<double>("scale_factor"));
    if (composed_attrs.Has("output_size")) {
      ctx->output_size = JUST(composed_attrs.GetAttr<std::vector<int64_t>>("output_size"));
    }
    ctx->data_format = JUST(composed_attrs.GetAttr<std::string>("data_format"));
    ctx->SaveTensorForBackward(inputs.at(0));
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const UpsampleNearest1DCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)
    const std::shared_ptr<oneflow::one::Tensor>& x = ctx->SavedTensors().at(0);
    in_grads->resize(1);
    JUST(oneflow::VectorAt(*in_grads, 0)) = JUST(
        functional::UpsampleNearest1DGrad(JUST(oneflow::VectorAt(out_grads, 0)), x,
                                          ctx->scale_factor, ctx->output_size, ctx->data_format));

    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("upsample_nearest_1d", UpsampleNearest1D);

struct UpsampleBicubic2DCaptureState : public AutoGradCaptureState {
  bool requires_grad = false;
  double height_scale = 0.0;
  double width_scale = 0.0;
  bool align_corners;
  std::vector<int64_t> output_size;
  std::string data_format;
};

class UpsampleBicubic2D : public OpExprGradFunction<UpsampleBicubic2DCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override { return Maybe<void>::Ok(); }

  Maybe<void> Capture(UpsampleBicubic2DCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);   // NOLINT(maybe-need-error-msg)
    CHECK_EQ_OR_RETURN(outputs.size(), 1);  // NOLINT(maybe-need-error-msg)
    ctx->requires_grad = inputs.at(0)->requires_grad();
    if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    ctx->height_scale = JUST(composed_attrs.GetAttr<double>("height_scale"));
    ctx->width_scale = JUST(composed_attrs.GetAttr<double>("width_scale"));
    ctx->align_corners = JUST(composed_attrs.GetAttr<bool>("align_corners"));
    if (composed_attrs.Has("output_size")) {
      ctx->output_size = JUST(composed_attrs.GetAttr<std::vector<int64_t>>("output_size"));
    }
    ctx->data_format = JUST(composed_attrs.GetAttr<std::string>("data_format"));
    ctx->SaveTensorForBackward(inputs.at(0));
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const UpsampleBicubic2DCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)
    const std::shared_ptr<oneflow::one::Tensor>& x = ctx->SavedTensors().at(0);
    in_grads->resize(1);
    JUST(oneflow::VectorAt(*in_grads, 0)) = JUST(functional::UpsampleBicubic2DGrad(
        JUST(oneflow::VectorAt(out_grads, 0)), x, ctx->height_scale, ctx->width_scale,
        ctx->align_corners, ctx->output_size, ctx->data_format));
    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("upsample_bicubic_2d", UpsampleBicubic2D);

struct UpsampleNearest3DCaptureState : public AutoGradCaptureState {
  bool requires_grad = false;
  double depth_scale = 0.0;
  double height_scale = 0.0;
  double width_scale = 0.0;
  std::vector<int64_t> output_size;
  std::string data_format;
};

class UpsampleNearest3D : public OpExprGradFunction<UpsampleNearest3DCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override { return Maybe<void>::Ok(); }

  Maybe<void> Capture(UpsampleNearest3DCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);   // NOLINT(maybe-need-error-msg)
    CHECK_EQ_OR_RETURN(outputs.size(), 1);  // NOLINT(maybe-need-error-msg)
    ctx->requires_grad = inputs.at(0)->requires_grad();
    if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    ctx->depth_scale = JUST(composed_attrs.GetAttr<double>("depth_scale"));
    ctx->height_scale = JUST(composed_attrs.GetAttr<double>("height_scale"));
    ctx->width_scale = JUST(composed_attrs.GetAttr<double>("width_scale"));
    if (composed_attrs.Has("output_size")) {
      ctx->output_size = JUST(composed_attrs.GetAttr<std::vector<int64_t>>("output_size"));
    }
    ctx->data_format = JUST(composed_attrs.GetAttr<std::string>("data_format"));
    ctx->SaveTensorForBackward(inputs.at(0));
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const UpsampleNearest3DCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)
    const std::shared_ptr<oneflow::one::Tensor>& x = ctx->SavedTensors().at(0);
    in_grads->resize(1);
    JUST(oneflow::VectorAt(*in_grads, 0)) = JUST(functional::UpsampleNearest3DGrad(
        JUST(oneflow::VectorAt(out_grads, 0)), x, ctx->depth_scale, ctx->height_scale,
        ctx->width_scale, ctx->output_size, ctx->data_format));

    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("upsample_nearest_3d", UpsampleNearest3D);

struct UpsampleTrilinear3DCaptureState : public AutoGradCaptureState {
  bool requires_grad = false;
  double depth_scale = 0.0;
  double height_scale = 0.0;
  double width_scale = 0.0;
  bool align_corners;
  std::vector<int64_t> output_size;
  std::string data_format;
};

class UpsampleTrilinear3D : public OpExprGradFunction<UpsampleTrilinear3DCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override { return Maybe<void>::Ok(); }

  Maybe<void> Capture(UpsampleTrilinear3DCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);   // NOLINT(maybe-need-error-msg)
    CHECK_EQ_OR_RETURN(outputs.size(), 1);  // NOLINT(maybe-need-error-msg)
    ctx->requires_grad = inputs.at(0)->requires_grad();
    if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    ctx->depth_scale = JUST(composed_attrs.GetAttr<double>("depth_scale"));
    ctx->height_scale = JUST(composed_attrs.GetAttr<double>("height_scale"));
    ctx->width_scale = JUST(composed_attrs.GetAttr<double>("width_scale"));
    ctx->align_corners = JUST(composed_attrs.GetAttr<bool>("align_corners"));
    if (composed_attrs.Has("output_size")) {
      ctx->output_size = JUST(composed_attrs.GetAttr<std::vector<int64_t>>("output_size"));
    }
    ctx->data_format = JUST(composed_attrs.GetAttr<std::string>("data_format"));
    ctx->SaveTensorForBackward(inputs.at(0));
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const UpsampleTrilinear3DCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)
    const std::shared_ptr<oneflow::one::Tensor>& x = ctx->SavedTensors().at(0);
    in_grads->resize(1);
    JUST(oneflow::VectorAt(*in_grads, 0)) = JUST(functional::UpsampleTrilinear3DGrad(
        JUST(oneflow::VectorAt(out_grads, 0)), x, ctx->depth_scale, ctx->height_scale,
        ctx->width_scale, ctx->align_corners, ctx->output_size, ctx->data_format));

    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("upsample_trilinear_3d", UpsampleTrilinear3D);

}  // namespace one
}  // namespace oneflow
