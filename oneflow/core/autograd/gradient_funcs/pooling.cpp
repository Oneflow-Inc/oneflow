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
  explicit PoolingNdGrad(const std::string& mode) : mode_(mode) {}
  virtual ~PoolingNdGrad() = default;

  Maybe<void> Capture(PoolingCaptureState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs, const OpBase* ctx) const override;
  Maybe<void> Apply(const PoolingCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  std::string mode_;
};

template<typename T>
Maybe<void> PoolingNdGrad<T>::Capture(PoolingCaptureState* state, const TensorTuple& inputs,
                                      const TensorTuple& outputs, const OpBase* ctx) const {
  state->requires_grad = inputs.at(0)->requires_grad();
  if (!state->requires_grad) { return Maybe<void>::Ok(); }

  state->input_index = state->SaveTensorForBackward(inputs.at(0));
  state->output_index = state->SaveTensorForBackward(outputs.at(0));
  state->indice_index = state->SaveTensorForBackward(outputs.at(1));

  auto* op_ctx = dynamic_cast<const typename T::OpT*>(ctx);
  state->data_format = op_ctx->data_format();
  state->padding = op_ctx->padding();
  state->kernel_size = op_ctx->kernel_size();
  state->stride = op_ctx->stride();
  state->dilation = op_ctx->dilation();
  state->return_indices = op_ctx->return_indices();
  state->ceil_mode = op_ctx->ceil_mode();
  return Maybe<void>::Ok();
}

template<typename T>
Maybe<void> PoolingNdGrad<T>::Apply(const PoolingCaptureState* state, const TensorTuple& out_grads,
                                    TensorTuple* in_grads) const {
  if (!state->requires_grad) { return Maybe<void>::Ok(); }
  CHECK_LE_OR_RETURN(out_grads.size(), 2);

  int32_t ndims = state->kernel_size.size();
  const auto& input = state->SavedTensors().at(state->input_index);
  const auto& output = state->SavedTensors().at(state->output_index);
  const auto& indice = state->SavedTensors().at(state->indice_index);

  in_grads->resize(1);
  in_grads->at(0) = JUST(functional::PoolingNdGrad(
      input, output, indice, out_grads.at(0), this->mode_, ndims, state->data_format,
      state->padding, state->kernel_size, state->stride, state->dilation, state->return_indices,
      state->ceil_mode));

  return Maybe<void>::Ok();
}

}  // namespace

class Maxpool1DGrad final : public PoolingNdGrad<Maxpool1DGrad> {
 public:
  using OpT = MaxPool1DGradOp;
  Maxpool1DGrad() : PoolingNdGrad<Maxpool1DGrad>("max") {}
};

class Maxpool2DGrad final : public PoolingNdGrad<Maxpool2DGrad> {
 public:
  using OpT = MaxPool2DGradOp;
  Maxpool2DGrad() : PoolingNdGrad<Maxpool2DGrad>("max") {}
};

class Maxpool3DGrad final : public PoolingNdGrad<Maxpool3DGrad> {
 public:
  using OpT = MaxPool3DGradOp;
  Maxpool3DGrad() : PoolingNdGrad<Maxpool3DGrad>("max") {}
};

REGISTER_OP_EXPR_GRAD_FUNCTION("maxpool_1d", Maxpool1DGrad);
REGISTER_OP_EXPR_GRAD_FUNCTION("maxpool_2d", Maxpool2DGrad);
REGISTER_OP_EXPR_GRAD_FUNCTION("maxpool_3d", Maxpool3DGrad);

}  // namespace one
}  // namespace oneflow
