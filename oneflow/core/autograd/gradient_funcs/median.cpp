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
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/functional/sequence_function.h"

namespace oneflow {
namespace one {

struct MedianCaptureState : public AutoGradCaptureState {
  bool requires_grad;
};

class Median : public OpExprGradFunction<MedianCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override { return Maybe<void>::Ok(); }
  Maybe<void> Capture(MedianCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);
    CHECK_EQ_OR_RETURN(outputs.size(), 1);
    ctx->requires_grad = inputs.at(0)->requires_grad();
    if (ctx->requires_grad) {
      ctx->SaveTensorForBackward(inputs.at(0));
      ctx->SaveTensorForBackward(outputs.at(0));
    }
    return Maybe<void>::Ok();
  }
  Maybe<void> Apply(const MedianCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    if (ctx->requires_grad) {
      const auto& input = ctx->SavedTensors().at(0);
      const auto& output = ctx->SavedTensors().at(1);
      const auto& dy = out_grads.at(0);
      std::vector<int32_t> axis(input->ndim());
      std::iota(axis.begin(), axis.end(), 0);
      const auto cast_like =
          JUST(functional::SequenceFunction<Maybe<Tensor>()>(
                   [&]() { return functional::BroadcastLike(output, input, axis); })
                   .then(std::bind(functional::BroadcastEqual, input, std::placeholders::_1))
                   .then(std::bind(functional::CastLike, std::placeholders::_1, input))
                   .call());

      const auto bcast_like_div =
          JUST(functional::SequenceFunction<Maybe<Tensor>()>(
                   [&]() { return functional::ReduceSum(cast_like, axis, false); })
                   .then(std::bind(functional::Div, dy, std::placeholders::_1))
                   .then(std::bind(functional::BroadcastLike, std::placeholders::_1, input, axis))
                   .call());

      in_grads->resize(1);
      in_grads->at(0) = JUST(functional::Mul(bcast_like_div, cast_like));
    }
    return Maybe<void>::Ok();
  }
};

struct MedianWithIndicesCaptureState : public AutoGradCaptureState {
  bool requires_grad;
};

class MedianWithIndices : public OpExprGradFunction<MedianWithIndicesCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override { return Maybe<void>::Ok(); }
  Maybe<void> Capture(MedianWithIndicesCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);
    CHECK_EQ_OR_RETURN(outputs.size(), 2);
    ctx->requires_grad = inputs.at(0)->requires_grad();
    if (ctx->requires_grad) { ctx->SaveTensorForBackward(outputs.at(1)); }
    return Maybe<void>::Ok();
  }
  Maybe<void> Apply(const MedianWithIndicesCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const {
    if (ctx->requires_grad) {
      in_grads->resize(1);
      const auto& indices = JUST(functional::Unsqueeze(ctx->SavedTensors().at(0), -1));
      const auto& dout = JUST(functional::Unsqueeze(out_grads.at(0), -1));
      JUST(functional::DimScatter(in_grads->at(0), -1, indices, dout));
    }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("median", Median);
REGISTER_OP_EXPR_GRAD_FUNCTION("median_with_indices", MedianWithIndices);

}  // namespace one
}  // namespace oneflow
