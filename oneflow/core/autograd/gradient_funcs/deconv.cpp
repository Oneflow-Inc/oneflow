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
};

class DeConvolutionNd : public OpExprGradFunction<DeConvolutionNdCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(DeConvolutionNdCaptureState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpInterpCtx* ctx) const override;
  Maybe<void> Apply(const DeConvolutionNdCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;
};

Maybe<void> DeConvolutionNd::Init(const OpExpr& op) {
  return Maybe<void>::Ok();
}

Maybe<void> DeConvolutionNd::Capture(DeConvolutionNdCaptureState* state, const TensorTuple& inputs,
                                     const TensorTuple& outputs, const OpInterpCtx* ctx) const {
  state->activation_requires_grad = inputs.at(0)->requires_grad();
  state->weight_requires_grad = inputs.at(1)->requires_grad();
  if (state->activation_requires_grad) {
    state->SaveTensorForBackward(inputs.at(1));  // weight
  }
  if (state->weight_requires_grad) {
    state->SaveTensorForBackward(inputs.at(0));  // x
  }

  auto* interp_ctx = dynamic_cast<const Deconv3DOpInterpCtx*>(ctx);
  state->data_format = interp_ctx->data_format;
  state->padding_before = interp_ctx->padding_before;
  state->kernel_size = interp_ctx->kernel_size;
  state->strides = interp_ctx->strides;
  state->dilation_rate = interp_ctx->dilation_rate;
  state->ndims = state->kernel_size.size();
  return Maybe<void>::Ok();
}

Maybe<void> DeConvolutionNd::Apply(const DeConvolutionNdCaptureState* state,
                                   const TensorTuple& out_grads, TensorTuple* in_grads) const {
  in_grads->resize(2);
  if (state->activation_requires_grad) {
    const auto& x = state->SavedTensors().at(1);
    std::vector<int64_t> start, stop, step;
    for (int i = 0; i < x->shape()->NumAxes(); i++) {
      start.push_back(0);
      stop.push_back(x->shape()->At(i));
      step.push_back(1);
    }
    const auto& weight = state->SavedTensors().at(0);
    if (state->ndims == 1) {
      std::shared_ptr<Tensor> result =
          JUST(functional::Conv1d(out_grads.at(0), weight, Optional<Tensor>(), state->strides,
                                  state->padding_before, state->dilation_rate, /*groups=*/1));
      result = JUST(functional::Slice(result, start, stop, step));
      in_grads->at(0) = result;
    } else if (state->ndims == 2) {
      std::shared_ptr<Tensor> result =
          JUST(functional::Conv2d(out_grads.at(0), weight, Optional<Tensor>(), state->strides,
                                  state->padding_before, state->dilation_rate, /*groups=*/1));
      result = JUST(functional::Slice(result, start, stop, step));
      in_grads->at(0) = result;
    } else if (state->ndims == 3) {
      std::shared_ptr<Tensor> result =
          JUST(functional::Conv3d(out_grads.at(0), weight, Optional<Tensor>(), state->strides,
                                  state->padding_before, state->dilation_rate, /*groups=*/1));
      result = JUST(functional::Slice(result, start, stop, step));
      in_grads->at(0) = result;
    } else {
      UNIMPLEMENTED_THEN_RETURN() << "Invalid ndim " << state->ndims << " for conv functor";
    }
  }
  if (state->weight_requires_grad) {
    int idx = state->activation_requires_grad;
    const auto& x = state->SavedTensors().at(idx);
    in_grads->at(1) = JUST(functional::ConvFilterGrad(
        x, out_grads.at(0), state->ndims, state->kernel_size, state->strides, state->padding_before,
        state->dilation_rate, /*groups=*/1, state->data_format));
  }
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("deconv1d", DeConvolutionNd);
REGISTER_OP_EXPR_GRAD_FUNCTION("deconv2d", DeConvolutionNd);
REGISTER_OP_EXPR_GRAD_FUNCTION("deconv3d", DeConvolutionNd);

}  // namespace one
}  // namespace oneflow
