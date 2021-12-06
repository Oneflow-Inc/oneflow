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
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

namespace {

struct PoolingCaptureState : public AutoGradCaptureState {
  bool requires_grad;
  size_t input_index;
  size_t output_index;
  size_t indice_index;

  std::string data_format;
  std::vector<int32_t> padding;
  std::vector<int32_t> kernel_size;
  std::vector<int32_t> stride;
  std::vector<int32_t> dilation;
  bool return_indices;
  bool ceil_mode;
};

template<typename T>
class PoolingNdGrad : public OpExprGradFunction<PoolingCaptureState> {
 public:
  virtual ~PoolingNdGrad() = default;

  using OpExprGradFunction<PoolingCaptureState>::Init;

  Maybe<void> Init(const OpExpr& op, const std::string& mode);
  Maybe<void> Capture(PoolingCaptureState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpInterpCtx* ctx) const override;
  Maybe<void> Apply(const PoolingCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  std::string mode_;
};

template<typename T>
Maybe<void> PoolingNdGrad::Init(const OpExpr& op, const std::string& mode) {
  const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);
  mode_ = mode;
  return Maybe<void>::Ok();
}

template<typename T>
Maybe<void> ::Capture(PoolingCaptureState* state, const TensorTuple& inputs,
                                   const TensorTuple& outputs, const OpInterpCtx* ctx) const {
  state->requires_grad = inputs.at(0)->requires_grad();
  if (!state->requires_grad) { return Maybe<void>::Ok(); }

  state->input_index = state->SaveTensorForBackward(inputs.at(0));
  state->output_index = state->SaveTensorForBackward(outputs.at(0));
  state->indice_index = state->SaveTensorForBackward(outputs.at(1));

  auto* interp_ctx = dynamic_cast<const T:ContextT*>(ctx);
  state->data_format = interp_ctx->data_format;
  state->padding = interp_ctx->padding;
  state->kernel_size = interp_ctx->kernel_size;
  state->stride = interp_ctx->stride;
  state->dilation = interp_ctx->dilation;
  state->return_indices = interp_ctx->return_indices;
  state->ceil_mode = interp_ctx->ceil_mode;
  return Maybe<void>::Ok();
}

template<typename T>
Maybe<void> PoolingNdGrad::Apply(const PoolingCaptureState* state, const TensorTuple& out_grads,
                                 TensorTuple* in_grads) const {
  if (!state->requires_grad) { return Maybe<void>::Ok(); }
  CHECK_LE_OR_RETURN(out_grads.size(), 2);

  int32_t ndims = state->kernel_size.size();
  const auto& input = state->SavedTensors().at(state->input_index);
  const auto& output = state->SavedTensors().at(state->output_index);
  const auto& indice = state->SavedTensors().at(state->indice_index);

  in_grads->resize(1);
  in_grads->at(0) = JUST(functional::PoolingNdGrad(
      input, output, indice, out_grads.at(0), mode_, ndims, state->data_format, state->padding,
      state->kernel_size, state->stride, state->dilation, state->return_indices, state->ceil_mode));

  return Maybe<void>::Ok();
}

}  // namespace


class Maxpool1DGrad final : public PoolingNdGrad<Maxpool1DGrad> {
 public:
  using ContextT = MaxPool1DGradOpInterpCtx;
  Maybe<void> Init(const OpExpr& op) override { return PoolingNdGrad::Init(op, "max"); }
};

class Maxpool2DGrad final : public PoolingNdGrad<Maxpool2DGrad> {
 public:
  using ContextT = MaxPool2DGradOpInterpCtx;
  Maybe<void> Init(const OpExpr& op) override { return PoolingNdGrad::Init(op, "max"); }
};

class Maxpool3DGrad final : public PoolingNdGrad<Maxpool3DGrad> {
 public:
  using ContextT = MaxPool3DGradOpInterpCtx;
  Maybe<void> Init(const OpExpr& op) override { return PoolingNdGrad::Init(op, "max"); }
};

class MaxpoolNdGrad final : public PoolingNdGrad {
 public:
  Maybe<void> Init(const OpExpr& op) override { return PoolingNdGrad::Init(op, "max"); }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("maxpool_1d", MaxpoolNdGrad);
REGISTER_OP_EXPR_GRAD_FUNCTION("maxpool_2d", MaxpoolNdGrad);
REGISTER_OP_EXPR_GRAD_FUNCTION("maxpool_3d", MaxpoolNdGrad);

}  // namespace one
}  // namespace oneflow
