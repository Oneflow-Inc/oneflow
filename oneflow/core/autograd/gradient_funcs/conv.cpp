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
#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/op_expr_grad_function.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct ConvolutionNdCaptureState : public AutoGradCaptureState {
  bool input_requires_grad = false;
  bool weight_requires_grad = false;
  bool has_bias = false;
  bool bias_requires_grad = false;
  size_t input_index;
  size_t weight_index;

  std::string data_format;
  std::vector<int32_t> padding_before;
  std::vector<int32_t> kernel_size;
  std::vector<int32_t> strides;
  std::vector<int32_t> dilation_rate;
  int32_t groups;
};

class ConvolutionNd : public OpExprGradFunction<ConvolutionNdCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(ConvolutionNdCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const ConvolutionNdCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
};

Maybe<void> ConvolutionNd::Init(const OpExpr& op) {
  const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  return Maybe<void>::Ok();
}

Maybe<void> ConvolutionNd::Capture(ConvolutionNdCaptureState* ctx, const TensorTuple& inputs,
                                   const TensorTuple& outputs, const AttrMap& attrs) const {
  CHECK_OR_RETURN(inputs.size() == 2 || inputs.size() == 3);  // NOLINT(maybe-need-error-msg)
  ctx->input_requires_grad = inputs.at(0)->requires_grad();
  ctx->weight_requires_grad = inputs.at(1)->requires_grad();
  if (inputs.size() == 3) {
    ctx->has_bias = true;
    ctx->bias_requires_grad = inputs.at(2)->requires_grad();
  }

  if (!ctx->input_requires_grad && !ctx->weight_requires_grad && !ctx->bias_requires_grad) {
    return Maybe<void>::Ok();
  }
  if (ctx->input_requires_grad) {
    ctx->weight_index = ctx->SaveTensorForBackward(inputs.at(1));  // weight
  }
  ctx->input_index = ctx->SaveTensorForBackward(inputs.at(0));  // input

  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->data_format = JUST(composed_attrs.GetAttr<std::string>("data_format"));
  ctx->padding_before = JUST(composed_attrs.GetAttr<std::vector<int32_t>>("padding_before"));
  ctx->kernel_size = JUST(composed_attrs.GetAttr<std::vector<int32_t>>("kernel_size"));
  ctx->strides = JUST(composed_attrs.GetAttr<std::vector<int32_t>>("strides"));
  ctx->dilation_rate = JUST(composed_attrs.GetAttr<std::vector<int32_t>>("dilation_rate"));
  ctx->groups = JUST(composed_attrs.GetAttr<int32_t>("groups"));
  return Maybe<void>::Ok();
}

Maybe<void> ConvolutionNd::Apply(const ConvolutionNdCaptureState* ctx, const TensorTuple& out_grads,
                                 TensorTuple* in_grads) const {
  if (ctx->has_bias) {
    in_grads->resize(3);
  } else {
    in_grads->resize(2);
  }
  size_t num_spatial_dims = ctx->kernel_size.size();
  if (ctx->input_requires_grad) {
    const auto& weight = ctx->SavedTensors().at(ctx->weight_index);
    const auto& input = ctx->SavedTensors().at(ctx->input_index);
    in_grads->at(0) = JUST(functional::ConvDataGrad(
        out_grads.at(0), weight, input, num_spatial_dims, ctx->kernel_size, ctx->strides,
        ctx->padding_before, ctx->dilation_rate, ctx->groups, ctx->data_format));
  }
  if (ctx->weight_requires_grad) {
    const auto& input = ctx->SavedTensors().at(ctx->input_index);
    in_grads->at(1) = JUST(functional::ConvFilterGrad(
        out_grads.at(0), input, num_spatial_dims, ctx->kernel_size, ctx->strides,
        ctx->padding_before, ctx->dilation_rate, ctx->groups, ctx->data_format));
  }
  if (ctx->bias_requires_grad) {
    std::vector<int32_t> dim;
    for (int i = 0; i < out_grads.at(0)->shape()->NumAxes(); ++i) {
      if ((ctx->data_format == "channels_first" && i == 1)
          || (ctx->data_format == "channels_last"
              && i == out_grads.at(0)->shape()->NumAxes() - 1)) {
        continue;
      }
      dim.push_back(i);
    }
    in_grads->at(2) = JUST(functional::ReduceSum(out_grads.at(0), dim, false));
  }
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("conv1d", ConvolutionNd);
REGISTER_OP_EXPR_GRAD_FUNCTION("conv2d", ConvolutionNd);
REGISTER_OP_EXPR_GRAD_FUNCTION("conv3d", ConvolutionNd);

}  // namespace one
}  // namespace oneflow
