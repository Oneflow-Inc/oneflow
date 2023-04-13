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
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/functional/impl/common.h"

namespace oneflow {
namespace one {

struct WhereCaptureState : public AutoGradCaptureState {
  bool requires_grad_x = false;
  bool requires_grad_y = false;
  DimVector x_reduce_dims = {};
  DimVector y_reduce_dims = {};
  DimVector x_squeeze_dims = {};
  DimVector y_squeeze_dims = {};
};

class Where : public OpExprGradFunction<WhereCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(WhereCaptureState* ctx, const TensorTuple& inputs, const TensorTuple& outputs,
                      const AttrMap& attrs) const override;
  Maybe<void> Apply(const WhereCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;
};

Maybe<void> Where::Init(const OpExpr& op) { return Maybe<void>::Ok(); }

Maybe<void> Where::Capture(WhereCaptureState* ctx, const TensorTuple& inputs,
                           const TensorTuple& outputs, const AttrMap& attrs) const {
  // cond, x, y
  CHECK_EQ_OR_RETURN(inputs.size(), 3);  // NOLINT(maybe-need-error-msg)
  ctx->requires_grad_x = inputs.at(1)->requires_grad();
  ctx->requires_grad_y = inputs.at(2)->requires_grad();
  if ((!ctx->requires_grad_x) && (!ctx->requires_grad_y)) { return Maybe<void>::Ok(); }

  ctx->SaveTensorForBackward(inputs.at(0));  // condition

  CHECK_EQ_OR_RETURN(outputs.size(), 1);
  const Shape& out_shape = *outputs.at(0)->shape();
  auto GetReduceDims = [&](DimVector& reduce_dim_vec, DimVector& squeeze_dim_vec,
                           const std::shared_ptr<oneflow::one::Tensor>& tensor) -> Maybe<void> {
    reduce_dim_vec.clear();
    squeeze_dim_vec.clear();
    const Shape& shape = *tensor->shape();
    if (functional::IsScalarTensor(tensor)) {
      reduce_dim_vec.resize(out_shape.size());
      squeeze_dim_vec.resize(out_shape.size());
      std::iota(reduce_dim_vec.begin(), reduce_dim_vec.end(), 0);
      std::iota(squeeze_dim_vec.begin(), squeeze_dim_vec.end(), 0);
    } else if (shape != out_shape) {
      CHECK_GE_OR_RETURN(out_shape.size(), shape.size());  // NOLINT(maybe-need-error-msg)
      size_t ddiff = out_shape.size() - shape.size();
      for (int i = 0; i < out_shape.size(); ++i) {
        if (i < ddiff) {
          reduce_dim_vec.push_back(i);
          squeeze_dim_vec.push_back(i);
        } else if (out_shape[i] != shape[i - ddiff]) {
          reduce_dim_vec.push_back(i);
        }
      }
    }
    return Maybe<void>::Ok();
  };
  JUST(GetReduceDims(ctx->x_reduce_dims, ctx->x_squeeze_dims, inputs.at(1)));
  JUST(GetReduceDims(ctx->y_reduce_dims, ctx->y_squeeze_dims, inputs.at(2)));

  return Maybe<void>::Ok();
}

Maybe<void> Where::Apply(const WhereCaptureState* ctx, const TensorTuple& out_grads,
                         TensorTuple* in_grads) const {
  if ((!ctx->requires_grad_x) && (!ctx->requires_grad_y)) { return Maybe<void>::Ok(); }
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)
  const auto& out_grad = out_grads.at(0);
  CHECK_EQ_OR_RETURN(ctx->SavedTensors().size(), 1);  // NOLINT(maybe-need-error-msg)
  const auto& condition = ctx->SavedTensors().at(0);
  std::shared_ptr<oneflow::one::Tensor> zero;
  if (out_grad->is_local()) {
    zero = JUST(
        functional::Constant(Shape({}), Scalar(0), out_grad->dtype(), JUST(out_grad->device())));
  } else {
    const size_t sbp_ndim = JUST(out_grad->nd_sbp())->sbp_parallel_size();
    std::vector<Symbol<SbpParallel>> nd_sbp_vec;
    nd_sbp_vec.reserve(sbp_ndim);
    for (int i = 0; i < sbp_ndim; ++i) {
      SbpParallel sbp;
      sbp.mutable_broadcast_parallel();
      nd_sbp_vec.push_back(SymbolOf(sbp));
    }
    const auto& parallel_desc = JUST(out_grad->parallel_desc());
    zero = JUST(functional::GlobalConstant(Shape({}), Scalar(0), out_grad->dtype(), parallel_desc,
                                           nd_sbp_vec));
  }
  in_grads->resize(3);  // cond, x, y
  if (ctx->requires_grad_x) {
    auto x_grad = JUST(functional::Where(condition, out_grad, zero));
    if (!ctx->x_reduce_dims.empty()) {
      x_grad = JUST(functional::ReduceSum(
          x_grad, std::vector<int32_t>{ctx->x_reduce_dims.begin(), ctx->x_reduce_dims.end()},
          /*keepdims=*/true));
    }
    if (!ctx->x_squeeze_dims.empty()) {
      x_grad = JUST(functional::Squeeze(
          x_grad, std::vector<int32_t>{ctx->x_squeeze_dims.begin(), ctx->x_squeeze_dims.end()}));
    }
    in_grads->at(1) = x_grad;
  }
  if (ctx->requires_grad_y) {
    auto y_grad = JUST(functional::Where(condition, zero, out_grad));
    if (!ctx->y_reduce_dims.empty()) {
      y_grad = JUST(functional::ReduceSum(
          y_grad, std::vector<int32_t>{ctx->y_reduce_dims.begin(), ctx->y_reduce_dims.end()},
          /*keepdims=*/true));
    }
    if (!ctx->y_squeeze_dims.empty()) {
      y_grad = JUST(functional::Squeeze(
          y_grad, std::vector<int32_t>{ctx->y_squeeze_dims.begin(), ctx->y_squeeze_dims.end()}));
    }
    in_grads->at(2) = y_grad;
  }
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("where", Where);

}  // namespace one
}  // namespace oneflow
