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
#include <cstdint>
#include "oneflow/core/common/optional.h"
#include "oneflow/core/framework/op_expr_grad_function.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct DeConvolutionNdCaptureState : public AutoGradCaptureState {
  bool weight_requires_grad = false;
  bool activation_requires_grad = false;
  size_t ndims;
  std::string data_format;
  std::vector<int32_t> padding_before;
  std::vector<int32_t> kernel_size;
  std::vector<int32_t> strides;
  std::vector<int32_t> dilation_rate;
  int32_t groups;
};

class DeConvolutionNd : public OpExprGradFunction<DeConvolutionNdCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(DeConvolutionNdCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const DeConvolutionNdCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
};

Maybe<void> DeConvolutionNd::Init(const OpExpr& op) {
  const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  return Maybe<void>::Ok();
}

Maybe<void> DeConvolutionNd::Capture(DeConvolutionNdCaptureState* ctx, const TensorTuple& inputs,
                                     const TensorTuple& outputs, const AttrMap& attrs) const {
  ctx->activation_requires_grad = inputs.at(0)->requires_grad();
  ctx->weight_requires_grad = inputs.at(1)->requires_grad();
  if (ctx->activation_requires_grad) {
    ctx->SaveTensorForBackward(inputs.at(1));  // weight
  }
  if (ctx->weight_requires_grad) {
    ctx->SaveTensorForBackward(inputs.at(0));  // x
  }

  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->data_format = JUST(composed_attrs.GetAttr<std::string>("data_format"));
  ctx->padding_before = JUST(composed_attrs.GetAttr<std::vector<int32_t>>("padding_before"));
  ctx->kernel_size = JUST(composed_attrs.GetAttr<std::vector<int32_t>>("kernel_size"));
  ctx->strides = JUST(composed_attrs.GetAttr<std::vector<int32_t>>("strides"));
  ctx->dilation_rate = JUST(composed_attrs.GetAttr<std::vector<int32_t>>("dilation_rate"));
  ctx->groups = JUST(composed_attrs.GetAttr<int32_t>("groups"));
  ctx->ndims = ctx->kernel_size.size();
  return Maybe<void>::Ok();
}

Maybe<void> DeConvolutionNd::Apply(const DeConvolutionNdCaptureState* ctx,
                                   const TensorTuple& out_grads, TensorTuple* in_grads) const {
  in_grads->resize(2);
  if (ctx->activation_requires_grad) {
    const auto& x = ctx->SavedTensors().at(1);
    std::vector<int64_t> start, stop, step;
    for (int i = 0; i < x->shape()->NumAxes(); i++) {
      start.emplace_back(0);
      stop.emplace_back(x->shape()->At(i));
      step.emplace_back(1);
    }
    const auto& weight = ctx->SavedTensors().at(0);
    if (ctx->ndims == 1) {
      std::shared_ptr<Tensor> result = JUST(functional::Conv1d(
          out_grads.at(0), weight, Optional<Tensor>(), ctx->strides, ctx->padding_before,
          ctx->dilation_rate, ctx->groups, ctx->data_format));
      result = JUST(functional::Slice(result, start, stop, step));
      in_grads->at(0) = result;
    } else if (ctx->ndims == 2) {
      std::shared_ptr<Tensor> result = JUST(functional::Conv2d(
          out_grads.at(0), weight, Optional<Tensor>(), ctx->strides, ctx->padding_before,
          ctx->dilation_rate, ctx->groups, ctx->data_format));
      result = JUST(functional::Slice(result, start, stop, step));
      in_grads->at(0) = result;
    } else if (ctx->ndims == 3) {
      std::shared_ptr<Tensor> result = JUST(functional::Conv3d(
          out_grads.at(0), weight, Optional<Tensor>(), ctx->strides, ctx->padding_before,
          ctx->dilation_rate, ctx->groups, ctx->data_format));
      result = JUST(functional::Slice(result, start, stop, step));
      in_grads->at(0) = result;
    } else {
      UNIMPLEMENTED_THEN_RETURN() << "Invalid ndim " << ctx->ndims << " for conv functor";
    }
  }
  if (ctx->weight_requires_grad) {
    int idx = ctx->activation_requires_grad;
    const auto& x = ctx->SavedTensors().at(idx);
    in_grads->at(1) = JUST(functional::ConvFilterGrad(
        x, out_grads.at(0), ctx->ndims, ctx->kernel_size, ctx->strides, ctx->padding_before,
        ctx->dilation_rate, ctx->groups, ctx->data_format));
  }
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("deconv1d", DeConvolutionNd);
REGISTER_OP_EXPR_GRAD_FUNCTION("deconv2d", DeConvolutionNd);
REGISTER_OP_EXPR_GRAD_FUNCTION("deconv3d", DeConvolutionNd);

}  // namespace one
}  // namespace oneflow
