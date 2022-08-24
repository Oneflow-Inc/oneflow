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

struct ConvDataGradGradCaptureState : public AutoGradCaptureState {
  bool w_requires_grad = false;
  bool grad_requires_grad = false;

  size_t w_index = 0;
  size_t grad_index = 0;

  std::string data_format;
  std::vector<int32_t> padding_before;
  std::vector<int32_t> kernel_size;
  std::vector<int32_t> strides;
  std::vector<int32_t> dilation_rate;
  int32_t groups = 0;
};

class ConvDataGradGrad : public OpExprGradFunction<ConvDataGradGradCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(ConvDataGradGradCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const ConvDataGradGradCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
};

Maybe<void> ConvDataGradGrad::Init(const OpExpr& op) {
  const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  return Maybe<void>::Ok();
}

Maybe<void> ConvDataGradGrad::Capture(ConvDataGradGradCaptureState* ctx, const TensorTuple& inputs,
                                      const TensorTuple& outputs, const AttrMap& attrs) const {
  // input: dy, w, x_like, [add to output]
  // output: dx
  CHECK_EQ_OR_RETURN(inputs.size(), 3);   // NOLINT(maybe-need-error-msg)
  CHECK_EQ_OR_RETURN(outputs.size(), 1);  // NOLINT(maybe-need-error-msg)

  ctx->w_requires_grad = inputs.at(1)->requires_grad();
  ctx->grad_requires_grad = inputs.at(0)->requires_grad();

  if (ctx->grad_requires_grad) { ctx->w_index = ctx->SaveTensorForBackward(inputs.at(1)); }
  if (ctx->w_requires_grad) { ctx->grad_index = ctx->SaveTensorForBackward(inputs.at(0)); }

  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->data_format = JUST(composed_attrs.GetAttr<std::string>("data_format"));
  ctx->padding_before = JUST(composed_attrs.GetAttr<std::vector<int32_t>>("padding_before"));
  ctx->kernel_size = JUST(composed_attrs.GetAttr<std::vector<int32_t>>("kernel_size"));
  ctx->strides = JUST(composed_attrs.GetAttr<std::vector<int32_t>>("strides"));
  ctx->dilation_rate = JUST(composed_attrs.GetAttr<std::vector<int32_t>>("dilation_rate"));
  ctx->groups = JUST(composed_attrs.GetAttr<int32_t>("groups"));
  return Maybe<void>::Ok();
}

Maybe<void> ConvDataGradGrad::Apply(const ConvDataGradGradCaptureState* ctx,
                                    const TensorTuple& out_grads, TensorTuple* in_grads) const {
  in_grads->resize(3);
  size_t num_spatial_dims = ctx->kernel_size.size();

  // first order forward: ConvND
  // x * w = y ( * => convolution)
  // first order backward:
  // x_grad = y_grad * w.rot180           (y.shape * w.shape -> x.shape)  call ConvDataGrad
  // w_grad = x * y_grad                  (x.shape * y.shape -> w.shape)  call ConvFilterGrad

  // second order forward (first order backward): ConvDataGrad
  // y_grad * w.rot180 = x_grad
  // second order forward:
  // w_grad_grad = out_grads_x * y_grad   (x.shape * y.shape -> w.shape)  call ConvFilterGrad
  // grad_for_y_grad = out_grads_x * w    (x.shape * w.shape -> y.shape)  call ConvND

  // w_grad_grad
  if (ctx->w_requires_grad) {
    const auto& grad = ctx->SavedTensors().at(ctx->grad_index);
    in_grads->at(1) = JUST(functional::ConvFilterGrad(
        grad, out_grads.at(0), num_spatial_dims, ctx->kernel_size, ctx->strides,
        ctx->padding_before, ctx->dilation_rate, ctx->groups, ctx->data_format));
  }

  // grad_for_y_grad
  if (ctx->grad_requires_grad) {
    const auto& w = ctx->SavedTensors().at(ctx->w_index);
    const int32_t ndims = ctx->kernel_size.size();
    const auto conv_op = (ndims == 1 ? functional::Conv1d
                                     : (ndims == 2 ? functional::Conv2d
                                                   : (ndims == 3 ? functional::Conv3d : nullptr)));
    CHECK_NOTNULL_OR_RETURN(conv_op);  // NOLINT(maybe-need-error-msg)
    in_grads->at(0) =
        JUST(conv_op(out_grads.at(0), w, Optional<Tensor>(), ctx->strides, ctx->padding_before,
                     ctx->dilation_rate, ctx->groups, ctx->data_format));
  }

  return Maybe<void>::Ok();
}

struct ConvFilterGradGradCaptureState : public AutoGradCaptureState {
  bool x_requires_grad = false;
  bool grad_requires_grad = false;

  size_t x_index = 0;
  size_t grad_index = 0;

  std::string data_format;
  std::vector<int32_t> padding_before;
  std::vector<int32_t> kernel_size;
  std::vector<int32_t> strides;
  std::vector<int32_t> dilation_rate;
  int32_t groups = 0;
};

class ConvFilterGradGrad : public OpExprGradFunction<ConvFilterGradGradCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(ConvFilterGradGradCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const ConvFilterGradGradCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
};

Maybe<void> ConvFilterGradGrad::Init(const OpExpr& op) {
  const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  return Maybe<void>::Ok();
}

Maybe<void> ConvFilterGradGrad::Capture(ConvFilterGradGradCaptureState* ctx,
                                        const TensorTuple& inputs, const TensorTuple& outputs,
                                        const AttrMap& attrs) const {
  // input: dy, x
  // output: dw
  CHECK_EQ_OR_RETURN(inputs.size(), 2);   // NOLINT(maybe-need-error-msg)
  CHECK_EQ_OR_RETURN(outputs.size(), 1);  // NOLINT(maybe-need-error-msg)

  ctx->x_requires_grad = inputs.at(1)->requires_grad();
  ctx->grad_requires_grad = inputs.at(0)->requires_grad();

  ctx->x_index = ctx->SaveTensorForBackward(inputs.at(1));
  if (ctx->x_requires_grad) { ctx->grad_index = ctx->SaveTensorForBackward(inputs.at(0)); }

  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->data_format = JUST(composed_attrs.GetAttr<std::string>("data_format"));
  ctx->padding_before = JUST(composed_attrs.GetAttr<std::vector<int32_t>>("padding_before"));
  ctx->kernel_size = JUST(composed_attrs.GetAttr<std::vector<int32_t>>("kernel_size"));
  ctx->strides = JUST(composed_attrs.GetAttr<std::vector<int32_t>>("strides"));
  ctx->dilation_rate = JUST(composed_attrs.GetAttr<std::vector<int32_t>>("dilation_rate"));
  ctx->groups = JUST(composed_attrs.GetAttr<int32_t>("groups"));
  return Maybe<void>::Ok();
}

Maybe<void> ConvFilterGradGrad::Apply(const ConvFilterGradGradCaptureState* ctx,
                                      const TensorTuple& out_grads, TensorTuple* in_grads) const {
  in_grads->resize(2);
  size_t num_spatial_dims = ctx->kernel_size.size();

  // first order forward: ConvND
  // x * w = y ( * => convolution)
  // first order backward:
  // x_grad = y_grad * w.rot180           (y.shape * w.shape -> x.shape)  call ConvDataGrad
  // w_grad = x * y_grad                  (x.shape * y.shape -> w.shape)  call ConvFilterGrad

  // second order forward (first order backward): ConvFilterGrad
  // x * y_grad = w_grad
  // second order backward:
  // x_grad_grad = out_grads_w * y_grad.rot180    (y.shape * w.shape -> x.shape)  call ConvDataGrad
  // grad_for_y_grad = x * out_grads_w            (x.shape * w.shape -> y.shape)  call ConvND

  // x_grad_grad
  if (ctx->x_requires_grad) {
    const auto& grad = ctx->SavedTensors().at(ctx->grad_index);
    const auto& x = ctx->SavedTensors().at(ctx->x_index);
    in_grads->at(1) = JUST(functional::ConvDataGrad(
        grad, out_grads.at(0), JUST(x->detach()), num_spatial_dims, ctx->kernel_size, ctx->strides,
        ctx->padding_before, ctx->dilation_rate, ctx->groups, ctx->data_format));
  }

  // grad_for_y_grad
  if (ctx->grad_requires_grad) {
    const auto& x = ctx->SavedTensors().at(ctx->x_index);
    const int32_t ndims = ctx->kernel_size.size();
    const auto conv_op = (ndims == 1 ? functional::Conv1d
                                     : (ndims == 2 ? functional::Conv2d
                                                   : (ndims == 3 ? functional::Conv3d : nullptr)));
    CHECK_NOTNULL_OR_RETURN(conv_op);  // NOLINT(maybe-need-error-msg)
    in_grads->at(0) =
        JUST(conv_op(x, out_grads.at(0), Optional<Tensor>(), ctx->strides, ctx->padding_before,
                     ctx->dilation_rate, ctx->groups, ctx->data_format));
  }

  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("conv_data_grad", ConvDataGradGrad);
REGISTER_OP_EXPR_GRAD_FUNCTION("conv_filter_grad", ConvFilterGradGrad);

}  // namespace one
}  // namespace oneflow
