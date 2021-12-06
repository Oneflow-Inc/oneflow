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

class PoolNdGrad : public OpExprGradFunction<PoolCaptureState> {
 public:
  virtual ~PoolNdGrad() = default;

  using OpExprGradFunction<PoolCaptureState>::Init;

  Maybe<void> Init(const OpExpr& op, const std::string& mode);
  Maybe<void> Capture(PoolCaptureState* state, const TensorTuple& inputs, const TensorTuple& outputs,
                      const OpInterpCtx* ctx) const override;
  Maybe<void> Apply(const PoolCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  std::string mode_;
};

Maybe<void> PoolNdGrad::Init(const OpExpr& op, const std::string& mode) {
  const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);
  mode_ = mode;
  return Maybe<void>::Ok();
}

Maybe<void> PoolNdGrad::Capture(PoolCaptureState* state, const TensorTuple& inputs,
                                const TensorTuple& outputs, const OpInterpCtx* ctx) const {
  state->requires_grad = inputs.at(0)->requires_grad();
  if (!state->requires_grad) { return Maybe<void>::Ok(); }

  state->input_index = state->SaveTensorForBackward(inputs.at(0));
  state->output_index = state->SaveTensorForBackward(outputs.at(0));

  auto* interp_ctx = dynamic_cast<const PLACEHOLDER*>(ctx);
  state->data_format = JUST(composed_attrs.GetAttr<std::string>("data_format"));
  state->padding = JUST(composed_attrs.GetAttr<std::string>("padding"));
  state->padding_before = JUST(composed_attrs.GetAttr<std::vector<int32_t>>("padding_before"));
  state->padding_after = JUST(composed_attrs.GetAttr<std::vector<int32_t>>("padding_after"));
  state->pool_size = JUST(composed_attrs.GetAttr<std::vector<int32_t>>("pool_size"));
  state->strides = JUST(composed_attrs.GetAttr<std::vector<int32_t>>("strides"));
  state->ceil_mode = JUST(composed_attrs.GetAttr<bool>("ceil_mode"));
  return Maybe<void>::Ok();
}

Maybe<void> PoolNdGrad::Apply(const PoolCaptureState* state, const TensorTuple& out_grads,
                              TensorTuple* in_grads) const {
  if (!state->requires_grad) { return Maybe<void>::Ok(); }
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);

  int32_t ndims = state->pool_size.size();
  const auto& input = state->SavedTensors().at(state->input_index);
  const auto& output = state->SavedTensors().at(state->output_index);

  in_grads->resize(1);
  in_grads->at(0) = JUST(functional::PoolNdGrad(
      input, output, out_grads.at(0), mode_, ndims, state->data_format, state->padding,
      state->padding_before, state->padding_after, state->pool_size, state->strides, state->ceil_mode));

  return Maybe<void>::Ok();
}

}  // namespace

class MaxPoolNdGrad final : public PoolNdGrad {
 public:
  Maybe<void> Init(const OpExpr& op) override { return PoolNdGrad::Init(op, "max"); }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("tf_max_pool_1d", MaxPoolNdGrad);
REGISTER_OP_EXPR_GRAD_FUNCTION("tf_max_pool_2d", MaxPoolNdGrad);
REGISTER_OP_EXPR_GRAD_FUNCTION("tf_max_pool_3d", MaxPoolNdGrad);

class AvgPoolNdGrad final : public PoolNdGrad {
 public:
  Maybe<void> Init(const OpExpr& op) override { return PoolNdGrad::Init(op, "avg"); }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("tf_avg_pool_1d", AvgPoolNdGrad);
REGISTER_OP_EXPR_GRAD_FUNCTION("tf_avg_pool_2d", AvgPoolNdGrad);
REGISTER_OP_EXPR_GRAD_FUNCTION("tf_avg_pool_3d", AvgPoolNdGrad);

}  // namespace one
}  // namespace oneflow
