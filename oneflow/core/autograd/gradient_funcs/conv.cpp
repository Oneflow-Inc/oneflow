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
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_generated.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct ConvolutionNdCaptureState : public AutoGradCaptureState {
  bool input_requires_grad = false;
  bool weight_requires_grad = false;
  size_t input_index;
  size_t weight_index;

  std::string data_format;
  std::vector<int32_t> padding_before;
  std::vector<int32_t> kernel_size;
  std::vector<int32_t> strides;
  std::vector<int32_t> dilation_rate;
  int32_t groups;
};

template<typename T>
class ConvolutionNd : public OpExprGradFunction<ConvolutionNdCaptureState> {
 public:
  Maybe<void> Capture(ConvolutionNdCaptureState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpBase* ctx) const override;
  Maybe<void> Apply(const ConvolutionNdCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;
};

template<typename T>
Maybe<void> ConvolutionNd<T>::Capture(ConvolutionNdCaptureState* state, const TensorTuple& inputs,
                                      const TensorTuple& outputs, const OpBase* ctx) const {
  CHECK_EQ_OR_RETURN(inputs.size(), 2);
  state->input_requires_grad = inputs.at(0)->requires_grad();
  state->weight_requires_grad = inputs.at(1)->requires_grad();
  if (!state->input_requires_grad && !state->weight_requires_grad) { return Maybe<void>::Ok(); }
  if (state->input_requires_grad) {
    state->weight_index = state->SaveTensorForBackward(inputs.at(1));  // weight
  }
  state->input_index = state->SaveTensorForBackward(inputs.at(0));  // input

  auto* op_ctx = JUST(ctx->dyn_cast<T>());
  state->data_format = op_ctx->data_format();
  state->padding_before = op_ctx->padding_before();
  state->kernel_size = op_ctx->kernel_size();
  state->strides = op_ctx->strides();
  state->dilation_rate = op_ctx->dilation_rate();
  state->groups = op_ctx->groups();
  return Maybe<void>::Ok();
}

template<typename T>
Maybe<void> ConvolutionNd<T>::Apply(const ConvolutionNdCaptureState* state,
                                    const TensorTuple& out_grads, TensorTuple* in_grads) const {
  in_grads->resize(2);
  size_t num_spatial_dims = state->kernel_size.size();
  if (state->input_requires_grad) {
    const auto& weight = state->SavedTensors().at(state->weight_index);
    const auto& input = state->SavedTensors().at(state->input_index);
    in_grads->at(0) = JUST(functional::ConvDataGrad(
        out_grads.at(0), weight, input, num_spatial_dims, state->kernel_size, state->strides,
        state->padding_before, state->dilation_rate, state->groups, state->data_format));
  }
  if (state->weight_requires_grad) {
    const auto& input = state->SavedTensors().at(state->input_index);
    in_grads->at(1) = JUST(functional::ConvFilterGrad(
        out_grads.at(0), input, num_spatial_dims, state->kernel_size, state->strides,
        state->padding_before, state->dilation_rate, state->groups, state->data_format));
  }
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("conv1d", ConvolutionNd<Conv1DOp>);
REGISTER_OP_EXPR_GRAD_FUNCTION("conv2d", ConvolutionNd<Conv2DOp>);
REGISTER_OP_EXPR_GRAD_FUNCTION("conv3d", ConvolutionNd<Conv3DOp>);

}  // namespace one
}  // namespace oneflow
