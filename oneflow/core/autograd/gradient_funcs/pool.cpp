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
#include "oneflow/core/framework/op_interp_ctx_generated.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

namespace {

struct PoolCaptureState : public AutoGradCaptureState {
  bool requires_grad;
  size_t input_index;
  size_t output_index;

  std::string data_format;
  std::string padding;
  std::vector<int32_t> padding_before;
  std::vector<int32_t> padding_after;
  std::vector<int32_t> pool_size;
  std::vector<int32_t> strides;
  bool ceil_mode;
};

template<typename T>
class PoolNdGrad : public OpExprGradFunction<PoolCaptureState> {
 public:
  explicit PoolNdGrad(const std::string& mode) : mode_(mode) {}
  virtual ~PoolNdGrad() = default;

  Maybe<void> Capture(PoolCaptureState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpInterpCtx* ctx) const override;
  Maybe<void> Apply(const PoolCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  std::string mode_;
};

template<typename T>
Maybe<void> PoolNdGrad<T>::Capture(PoolCaptureState* state, const TensorTuple& inputs,
                                   const TensorTuple& outputs, const OpInterpCtx* ctx) const {
  state->requires_grad = inputs.at(0)->requires_grad();
  if (!state->requires_grad) { return Maybe<void>::Ok(); }

  state->input_index = state->SaveTensorForBackward(inputs.at(0));
  state->output_index = state->SaveTensorForBackward(outputs.at(0));

  auto* interp_ctx = dynamic_cast<const typename T::ContextT*>(ctx);
  state->data_format = interp_ctx->data_format();
  state->padding = interp_ctx->padding();
  state->padding_before = interp_ctx->padding_before();
  state->padding_after = interp_ctx->padding_after();
  state->pool_size = interp_ctx->pool_size();
  state->strides = interp_ctx->strides();
  state->ceil_mode = interp_ctx->ceil_mode();
  return Maybe<void>::Ok();
}

template<typename T>
Maybe<void> PoolNdGrad<T>::Apply(const PoolCaptureState* state, const TensorTuple& out_grads,
                                 TensorTuple* in_grads) const {
  if (!state->requires_grad) { return Maybe<void>::Ok(); }
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);

  int32_t ndims = state->pool_size.size();
  const auto& input = state->SavedTensors().at(state->input_index);
  const auto& output = state->SavedTensors().at(state->output_index);

  in_grads->resize(1);
  in_grads->at(0) = JUST(
      functional::PoolNdGrad(input, output, out_grads.at(0), this->mode_, ndims, state->data_format,
                             state->padding, state->padding_before, state->padding_after,
                             state->pool_size, state->strides, state->ceil_mode));

  return Maybe<void>::Ok();
}

}  // namespace

class MaxPool1DGrad : public PoolNdGrad<MaxPool1DGrad> {
 public:
  using ContextT = TfMaxPool1DOpInterpCtx;
  MaxPool1DGrad() : PoolNdGrad<MaxPool1DGrad>("max") {}
};

class MaxPool2DGrad : public PoolNdGrad<MaxPool2DGrad> {
 public:
  using ContextT = TfMaxPool2DOpInterpCtx;
  MaxPool2DGrad() : PoolNdGrad<MaxPool2DGrad>("max") {}
};

class MaxPool3DGrad : public PoolNdGrad<MaxPool3DGrad> {
 public:
  using ContextT = TfMaxPool3DOpInterpCtx;
  MaxPool3DGrad() : PoolNdGrad<MaxPool3DGrad>("max") {}
};

REGISTER_OP_EXPR_GRAD_FUNCTION("tf_max_pool_1d", MaxPool1DGrad);
REGISTER_OP_EXPR_GRAD_FUNCTION("tf_max_pool_2d", MaxPool2DGrad);
REGISTER_OP_EXPR_GRAD_FUNCTION("tf_max_pool_3d", MaxPool3DGrad);

class AvgPool1DGrad : public PoolNdGrad<AvgPool1DGrad> {
 public:
  using ContextT = TfAvgPool1DOpInterpCtx;
  AvgPool1DGrad() : PoolNdGrad<AvgPool1DGrad>("avg") {}
};

class AvgPool2DGrad : public PoolNdGrad<AvgPool2DGrad> {
 public:
  using ContextT = TfAvgPool2DOpInterpCtx;
  AvgPool2DGrad() : PoolNdGrad<AvgPool2DGrad>("avg") {}
};

class AvgPool3DGrad : public PoolNdGrad<AvgPool3DGrad> {
 public:
  using ContextT = TfAvgPool3DOpInterpCtx;
  AvgPool3DGrad() : PoolNdGrad<AvgPool3DGrad>("avg") {}
};

REGISTER_OP_EXPR_GRAD_FUNCTION("tf_avg_pool_1d", AvgPool1DGrad);
REGISTER_OP_EXPR_GRAD_FUNCTION("tf_avg_pool_2d", AvgPool2DGrad);
REGISTER_OP_EXPR_GRAD_FUNCTION("tf_avg_pool_3d", AvgPool3DGrad);

}  // namespace one
}  // namespace oneflow
