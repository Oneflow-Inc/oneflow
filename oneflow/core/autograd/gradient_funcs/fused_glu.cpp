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
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/framework/op_expr_grad_function.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct FusedGluGradCaptureState : public AutoGradCaptureState {
  bool is_split_mode = false;
  bool has_bias = false;
  std::string activation = "none";
  bool w_requires_grad = false;
  bool v_requires_grad = false;
  bool b_requires_grad = false;
  bool c_requires_grad = false;
};

class FusedGluGrad : public OpExprGradFunction<FusedGluGradCaptureState> {
  Maybe<void> Init(const OpExpr& op) override;

  Maybe<void> Capture(FusedGluGradCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;

  Maybe<void> Apply(const FusedGluGradCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
};

Maybe<void> FusedGluGrad::Init(const OpExpr& op) {
  const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  return Maybe<void>::Ok();
}

Maybe<void> FusedGluGrad::Capture(FusedGluGradCaptureState* ctx, const TensorTuple& inputs,
                                  const TensorTuple& outputs, const AttrMap& attrs) const {
  // check input size
  const size_t in_size = inputs.size();
  CHECK_OR_RETURN(in_size == 2 || in_size == 3 || in_size == 5)
      << "FusedGluGrad::Capture(): input tensor size must be 2 or 3 or 5";

  // check the input pattern:
  ctx->has_bias = JUST(attrs.GetAttr<bool>("has_bias"));
  ctx->is_split_mode = JUST(attrs.GetAttr<bool>("is_split"));

  // check whether input tensors need grad
  ctx->w_requires_grad = inputs[1]->requires_grad();
  if (ctx->has_bias) {
    ctx->b_requires_grad = inputs[2]->requires_grad();
    if (ctx->is_split_mode) {
      ctx->v_requires_grad = inputs[3]->requires_grad();
      ctx->c_requires_grad = inputs[4]->requires_grad();
    }
  } else {
    if (ctx->is_split_mode) { ctx->v_requires_grad = inputs[2]->requires_grad(); }
  }

  // save tensors for backward
  ctx->SaveTensorForBackward(inputs[0]);   // x
  ctx->SaveTensorForBackward(outputs[1]);  // matmul_wx
  if (ctx->is_split_mode) {
    ctx->SaveTensorForBackward(outputs[2]);  // matmul_vx
  }

  // save activation type
  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->activation = JUST(composed_attrs.GetAttr<std::string>("activation"));

  return Maybe<void>::Ok();
}

Maybe<void> FusedGluGrad::Apply(const FusedGluGradCaptureState* ctx, const TensorTuple& out_grads,
                                TensorTuple* in_grads) const {
  // obtain saved tensors from forward process
  const auto& x = ctx->SavedTensors()[0];
  const auto& matmul_wx = ctx->SavedTensors()[1];

  // obtain gradient dy
  const auto& dy = out_grads[0];

  if (ctx->is_split_mode) {
    // obtain saved optional tensor from forward process
    const auto& matmul_vx = ctx->SavedTensors()[2];

    if (ctx->w_requires_grad or ctx->b_requires_grad or ctx->v_requires_grad
        or ctx->c_requires_grad) {
      // calculate the intermediate gradient using fused kernel
      const auto& middle_results =
          JUST(functional::FusedGluWithoutLinearGrad(dy, matmul_wx, matmul_vx, ctx->activation));
      const auto& d_matmul_wx = (*middle_results)[0];
      const auto& d_matmul_vx = (*middle_results)[1];

      // calculate the final gradient result of w (if necessary)
      if (ctx->w_requires_grad) {
        (*in_grads)[1] = JUST(functional::BroadcastMatmulGradB(d_matmul_wx, x, 1.0));
      }

      // calculate the final gradient result of b (if necessary)
      if (ctx->b_requires_grad) {
        const int64_t num_axes = d_matmul_wx->shape()->NumAxes();
        std::vector<int32_t> reduce_axes_vec;
        reduce_axes_vec.reserve(num_axes - 1);
        for (int i = 0; i < num_axes - 1; i++) { reduce_axes_vec.push_back(i); }

        (*in_grads)[2] = JUST(functional::ReduceSum(d_matmul_wx, reduce_axes_vec, false));
      }

      // calculate the final gradient result of v (if necessary)
      if (ctx->v_requires_grad) {
        if (ctx->has_bias) {
          (*in_grads)[3] = JUST(functional::BroadcastMatmulGradB(d_matmul_vx, x, 1.0));
        } else {
          (*in_grads)[2] = JUST(functional::BroadcastMatmulGradB(d_matmul_vx, x, 1.0));
        }
      }

      // calculate the final gradient result of c (if necessary)
      if (ctx->c_requires_grad) {
        const int64_t num_axes = d_matmul_vx->shape()->NumAxes();
        std::vector<int32_t> reduce_axes_vec;
        reduce_axes_vec.reserve(num_axes - 1);
        for (int i = 0; i < num_axes - 1; i++) { reduce_axes_vec.push_back(i); }

        (*in_grads)[4] = JUST(functional::ReduceSum(d_matmul_vx, reduce_axes_vec, false));
      }
    }
  } else {
    if (ctx->w_requires_grad or ctx->b_requires_grad) {
      // calculate the intermediate gradient using fused kernel
      const auto& middle_results =
          JUST(functional::FusedGluWithoutLinearGrad(dy, matmul_wx, nullptr, ctx->activation));
      const auto& d_matmul_wx = (*middle_results)[0];

      // calculate the final gradient result of w (if necessary)
      if (ctx->w_requires_grad) {
        (*in_grads)[1] = JUST(functional::BroadcastMatmulGradB(d_matmul_wx, x, 1.0));
      }

      // calculate the final gradient result of b (if necessary)
      if (ctx->b_requires_grad) {
        const int64_t num_axes = d_matmul_wx->shape()->NumAxes();
        std::vector<int32_t> reduce_axes_vec;
        reduce_axes_vec.reserve(num_axes - 1);
        for (int i = 0; i < num_axes - 1; i++) { reduce_axes_vec.push_back(i); }

        (*in_grads)[2] = JUST(functional::ReduceSum(d_matmul_wx, reduce_axes_vec, false));
      }
    }
  }

  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("fused_glu", FusedGluGrad);

}  // namespace one
}  // namespace oneflow
