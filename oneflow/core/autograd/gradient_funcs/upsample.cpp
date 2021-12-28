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
#include "oneflow/core/framework/op_generated.h"

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
  Maybe<void> Capture(UpsampleCaptureState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpBase* ctx) const override;
  Maybe<void> Apply(const UpsampleCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;
};

Maybe<void> Upsample::Capture(UpsampleCaptureState* state, const TensorTuple& inputs,
                              const TensorTuple& outputs, const OpBase* ctx) const {
  state->requires_grad = inputs.at(0)->requires_grad();
  if (!state->requires_grad) { return Maybe<void>::Ok(); }
  auto* op_ctx = dynamic_cast<const UpsampleOp*>(ctx);
  state->height_scale = op_ctx->height_scale();
  state->width_scale = op_ctx->width_scale();
  state->align_corners = op_ctx->align_corners();
  state->data_format = op_ctx->data_format();
  state->interpolation = op_ctx->interpolation();
  state->SaveTensorForBackward(inputs.at(0));
  return Maybe<void>::Ok();
}

Maybe<void> Upsample::Apply(const UpsampleCaptureState* state, const TensorTuple& out_grads,
                            TensorTuple* in_grads) const {
  if (!state->requires_grad) { return Maybe<void>::Ok(); }
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);

  const std::shared_ptr<oneflow::one::Tensor>& x = state->SavedTensors().at(0);
  in_grads->resize(1);
  in_grads->at(0) = JUST(functional::UpsampleGrad(out_grads.at(0), x, state->height_scale,
                                                  state->width_scale, state->align_corners,
                                                  state->data_format, state->interpolation));
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
  Maybe<void> Capture(UpsampleNearest2DCaptureState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpBase* ctx) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);
    CHECK_EQ_OR_RETURN(outputs.size(), 1);
    state->requires_grad = inputs.at(0)->requires_grad();
    if (!state->requires_grad) { return Maybe<void>::Ok(); }
    auto* op_ctx = dynamic_cast<const UpsampleNearest2DOp*>(ctx);
    state->height_scale = op_ctx->height_scale();
    state->width_scale = op_ctx->width_scale();
    state->data_format = op_ctx->data_format();
    state->SaveTensorForBackward(inputs.at(0));
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const UpsampleNearest2DCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    if (!state->requires_grad) { return Maybe<void>::Ok(); }
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    const std::shared_ptr<oneflow::one::Tensor>& x = state->SavedTensors().at(0);
    in_grads->resize(1);
    in_grads->at(0) = JUST(functional::UpsampleNearest2DGrad(
        out_grads.at(0), x, state->height_scale, state->width_scale, state->data_format));

    return Maybe<void>::Ok();
  }
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
  Maybe<void> Capture(UpsampleBilinear2DCaptureState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpBase* ctx) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);
    CHECK_EQ_OR_RETURN(outputs.size(), 1);
    state->requires_grad = inputs.at(0)->requires_grad();
    if (!state->requires_grad) { return Maybe<void>::Ok(); }
    auto* op_ctx = dynamic_cast<const UpsampleBilinear2DOp*>(ctx);
    state->height_scale = op_ctx->height_scale();
    state->width_scale = op_ctx->width_scale();
    state->align_corners = op_ctx->align_corners();
    state->data_format = op_ctx->data_format();
    state->SaveTensorForBackward(inputs.at(0));
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const UpsampleBilinear2DCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    if (!state->requires_grad) { return Maybe<void>::Ok(); }
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    const std::shared_ptr<oneflow::one::Tensor>& x = state->SavedTensors().at(0);
    in_grads->resize(1);
    in_grads->at(0) = JUST(functional::UpsampleBilinear2DGrad(
        out_grads.at(0), x, state->height_scale, state->width_scale, state->align_corners,
        state->data_format));

    return Maybe<void>::Ok();
  }
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
  Maybe<void> Capture(UpsampleLinear1DCaptureState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpBase* ctx) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);
    CHECK_EQ_OR_RETURN(outputs.size(), 1);
    state->requires_grad = inputs.at(0)->requires_grad();
    if (!state->requires_grad) { return Maybe<void>::Ok(); }
    auto* op_ctx = dynamic_cast<const UpsampleLinear1DOp*>(ctx);
    state->scale_factor = op_ctx->scale_factor();
    state->align_corners = op_ctx->align_corners();
    state->data_format = op_ctx->data_format();
    state->SaveTensorForBackward(inputs.at(0));
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const UpsampleLinear1DCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    if (!state->requires_grad) { return Maybe<void>::Ok(); }
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    const std::shared_ptr<oneflow::one::Tensor>& x = state->SavedTensors().at(0);
    in_grads->resize(1);
    in_grads->at(0) = JUST(functional::UpsampleLinear1DGrad(
        out_grads.at(0), x, state->scale_factor, state->align_corners, state->data_format));

    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("upsample_linear_1d", UpsampleLinear1D);

struct UpsampleNearest1DCaptureState : public AutoGradCaptureState {
  bool requires_grad;
  float scale_factor;
  std::string data_format;
};

class UpsampleNearest1D : public OpExprGradFunction<UpsampleNearest1DCaptureState> {
 public:
  Maybe<void> Capture(UpsampleNearest1DCaptureState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpBase* ctx) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);
    CHECK_EQ_OR_RETURN(outputs.size(), 1);
    state->requires_grad = inputs.at(0)->requires_grad();
    if (!state->requires_grad) { return Maybe<void>::Ok(); }
    auto* op_ctx = dynamic_cast<const UpsampleNearest1DOp*>(ctx);
    state->scale_factor = op_ctx->scale_factor();
    state->data_format = op_ctx->data_format();
    state->SaveTensorForBackward(inputs.at(0));
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const UpsampleNearest1DCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    if (!state->requires_grad) { return Maybe<void>::Ok(); }
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    const std::shared_ptr<oneflow::one::Tensor>& x = state->SavedTensors().at(0);
    in_grads->resize(1);
    in_grads->at(0) = JUST(functional::UpsampleNearest1DGrad(
        out_grads.at(0), x, state->scale_factor, state->data_format));

    return Maybe<void>::Ok();
  }
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
  Maybe<void> Capture(UpsampleBicubic2DCaptureState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpBase* ctx) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);
    CHECK_EQ_OR_RETURN(outputs.size(), 1);
    state->requires_grad = inputs.at(0)->requires_grad();
    if (!state->requires_grad) { return Maybe<void>::Ok(); }
    auto* op_ctx = dynamic_cast<const UpsampleBicubic2DOp*>(ctx);
    state->height_scale = op_ctx->height_scale();
    state->width_scale = op_ctx->width_scale();
    state->align_corners = op_ctx->align_corners();
    state->data_format = op_ctx->data_format();
    state->SaveTensorForBackward(inputs.at(0));
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const UpsampleBicubic2DCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    if (!state->requires_grad) { return Maybe<void>::Ok(); }
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    const std::shared_ptr<oneflow::one::Tensor>& x = state->SavedTensors().at(0);
    in_grads->resize(1);
    in_grads->at(0) = JUST(functional::UpsampleBicubic2DGrad(
        out_grads.at(0), x, state->height_scale, state->width_scale, state->align_corners,
        state->data_format));
    return Maybe<void>::Ok();
  }
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
  Maybe<void> Capture(UpsampleNearest3DCaptureState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpBase* ctx) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);
    CHECK_EQ_OR_RETURN(outputs.size(), 1);
    state->requires_grad = inputs.at(0)->requires_grad();
    if (!state->requires_grad) { return Maybe<void>::Ok(); }
    auto* op_ctx = dynamic_cast<const UpsampleNearest3DOp*>(ctx);
    state->depth_scale = op_ctx->depth_scale();
    state->height_scale = op_ctx->height_scale();
    state->width_scale = op_ctx->width_scale();
    state->data_format = op_ctx->data_format();
    state->SaveTensorForBackward(inputs.at(0));
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const UpsampleNearest3DCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    if (!state->requires_grad) { return Maybe<void>::Ok(); }
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    const std::shared_ptr<oneflow::one::Tensor>& x = state->SavedTensors().at(0);
    in_grads->resize(1);
    in_grads->at(0) = JUST(functional::UpsampleNearest3DGrad(
        out_grads.at(0), x, state->depth_scale, state->height_scale, state->width_scale,
        state->data_format));

    return Maybe<void>::Ok();
  }
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
  Maybe<void> Capture(UpsampleTrilinear3DCaptureState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpBase* ctx) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);
    CHECK_EQ_OR_RETURN(outputs.size(), 1);
    state->requires_grad = inputs.at(0)->requires_grad();
    if (!state->requires_grad) { return Maybe<void>::Ok(); }
    auto* op_ctx = dynamic_cast<const UpsampleTrilinear3DOp*>(ctx);
    state->depth_scale = op_ctx->depth_scale();
    state->height_scale = op_ctx->height_scale();
    state->width_scale = op_ctx->width_scale();
    state->align_corners = op_ctx->align_corners();
    state->data_format = op_ctx->data_format();
    state->SaveTensorForBackward(inputs.at(0));
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const UpsampleTrilinear3DCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    if (!state->requires_grad) { return Maybe<void>::Ok(); }
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    const std::shared_ptr<oneflow::one::Tensor>& x = state->SavedTensors().at(0);
    in_grads->resize(1);
    in_grads->at(0) = JUST(functional::UpsampleTrilinear3DGrad(
        out_grads.at(0), x, state->depth_scale, state->height_scale, state->width_scale,
        state->align_corners, state->data_format));

    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("upsample_trilinear_3d", UpsampleTrilinear3D);

}  // namespace one
}  // namespace oneflow
