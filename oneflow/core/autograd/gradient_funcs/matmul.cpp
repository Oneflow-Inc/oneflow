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
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/common/container_util.h"

namespace oneflow {
namespace one {

struct MatmulCaptureState : public AutoGradCaptureState {
  bool transpose_a;
  bool transpose_b;
  double alpha;
  bool requires_grad_a;
  bool requires_grad_b;
  size_t a_index;
  size_t b_index;
};

class Matmul : public OpExprGradFunction<MatmulCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(MatmulCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const MatmulCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 protected:
  AttrMap base_attrs_;
};

Maybe<void> Matmul::Init(const OpExpr& op) {
  const UserOpExpr* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());

  return Maybe<void>::Ok();
}

Maybe<void> Matmul::Capture(MatmulCaptureState* ctx, const TensorTuple& inputs,
                            const TensorTuple& outputs, const AttrMap& attrs) const {
  ctx->requires_grad_a = inputs.at(0)->requires_grad();
  ctx->requires_grad_b = inputs.at(1)->requires_grad();
  if (!ctx->requires_grad_a && !ctx->requires_grad_b) { return Maybe<void>::Ok(); }

  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->transpose_a = JUST(composed_attrs.GetAttr<bool>("transpose_a"));
  ctx->transpose_b = JUST(composed_attrs.GetAttr<bool>("transpose_b"));
  ctx->alpha = JUST(composed_attrs.GetAttr<double>("alpha"));
  if (ctx->requires_grad_a) {
    ctx->b_index = ctx->SaveTensorForBackward(inputs.at(1));  // input b
  }
  if (ctx->requires_grad_b) {
    ctx->a_index = ctx->SaveTensorForBackward(inputs.at(0));  // input a
  }
  return Maybe<void>::Ok();
}

Maybe<void> Matmul::Apply(const MatmulCaptureState* ctx, const TensorTuple& out_grads,
                          TensorTuple* in_grads) const {
  if (!ctx->requires_grad_a && !ctx->requires_grad_b) { return Maybe<void>::Ok(); }
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)

  in_grads->resize(2);
  if (ctx->requires_grad_a) {
    const auto& input_b = ctx->SavedTensors().at(ctx->b_index);
    if (ctx->transpose_a) {
      in_grads->at(0) =
          JUST(functional::MatMul(input_b, out_grads.at(0), ctx->transpose_b, true, ctx->alpha));
    } else {
      in_grads->at(0) = JUST(
          functional::MatMul(out_grads.at(0), input_b, false, !(ctx->transpose_b), ctx->alpha));
    }
  }

  if (ctx->requires_grad_b) {
    const auto& input_a = ctx->SavedTensors().at(ctx->a_index);
    if (ctx->transpose_b) {
      in_grads->at(1) =
          JUST(functional::MatMul(out_grads.at(0), input_a, true, ctx->transpose_a, ctx->alpha));
    } else {
      in_grads->at(1) = JUST(
          functional::MatMul(input_a, out_grads.at(0), !(ctx->transpose_a), false, ctx->alpha));
    }
  }

  return Maybe<void>::Ok();
}

struct BroadcastMatmulCaptureState : public AutoGradCaptureState {
  bool transpose_a = false;
  bool transpose_b = false;
  double alpha = 1.0;
  bool requires_grad_a = true;
  bool requires_grad_b = true;
  size_t a_index = 0;
  size_t b_index = 1;
  bool broadcast_a = false;
  bool broadcast_b = false;
  int64_t b_num_axes = 0;
};

class BroadcastMatmul : public OpExprGradFunction<BroadcastMatmulCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(BroadcastMatmulCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const BroadcastMatmulCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 protected:
  AttrMap base_attrs_;
};

Maybe<void> BroadcastMatmul::Init(const OpExpr& op) {
  const UserOpExpr* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr) << "fw_op_expr should not be null. ";
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());

  return Maybe<void>::Ok();
}

Maybe<void> BroadcastMatmul::Capture(BroadcastMatmulCaptureState* ctx, const TensorTuple& inputs,
                                     const TensorTuple& outputs, const AttrMap& attrs) const {
  ctx->requires_grad_a = JUST(VectorAt(inputs, 0))->requires_grad();
  ctx->requires_grad_b = JUST(VectorAt(inputs, 1))->requires_grad();
  if (!ctx->requires_grad_a && !ctx->requires_grad_b) { return Maybe<void>::Ok(); }

  const auto a_shape = JUST(VectorAt(inputs, 0))->shape();
  const auto b_shape = JUST(VectorAt(inputs, 1))->shape();

  const int64_t a_num_axes = a_shape->NumAxes();
  const int64_t b_num_axes = b_shape->NumAxes();

  const size_t num_max_batch_dims = std::max(a_num_axes, b_num_axes) - 2;
  auto MakeGetBatchDim = [num_max_batch_dims](size_t num_dims, const Shape& shape_dim) {
    const int64_t num_batch_dims = num_dims - 2;
    const int64_t num_padding_dims = num_max_batch_dims - num_batch_dims;
    return [num_padding_dims, shape_dim](size_t index) {
      return index < num_padding_dims ? 1 : shape_dim.At(index - num_padding_dims);
    };
  };
  auto GetABatchDim = MakeGetBatchDim(a_num_axes, *a_shape);
  auto GetBBatchDim = MakeGetBatchDim(b_num_axes, *b_shape);
  bool broadcast_a = false;
  bool broadcast_b = false;

  for (int32_t i = 0; i < num_max_batch_dims; i++) {
    if (GetABatchDim(i) < GetBBatchDim(i) || a_num_axes < b_num_axes) {
      broadcast_a = true;
      break;
    }
  }

  for (int32_t i = 0; i < num_max_batch_dims; i++) {
    if (GetBBatchDim(i) < GetABatchDim(i) || b_num_axes < a_num_axes) {
      broadcast_b = true;
      break;
    }
  }

  if (b_num_axes == 2 && !ctx->transpose_a) {
    // In this case, we can directly use `broadcast_matmul_grad_b` OP to generate Grad instead of
    // broadcast_matmul+reduce_sum_like.
    broadcast_b = false;
  }

  ctx->broadcast_a = broadcast_a;
  ctx->broadcast_b = broadcast_b;

  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->transpose_a = JUST(composed_attrs.GetAttr<bool>("transpose_a"));
  ctx->transpose_b = JUST(composed_attrs.GetAttr<bool>("transpose_b"));
  ctx->alpha = JUST(composed_attrs.GetAttr<double>("alpha"));

  if (ctx->requires_grad_a) {
    ctx->b_index = ctx->SaveTensorForBackward(JUST(VectorAt(inputs, 1)));  // input b
    if (broadcast_a) {
      ctx->a_index = ctx->SaveTensorForBackward(JUST(VectorAt(inputs, 0)));  // input a
    }
  }

  if (ctx->requires_grad_b) {
    ctx->b_num_axes = JUST(VectorAt(inputs, 1))->shape()->NumAxes();
    ctx->a_index = ctx->SaveTensorForBackward(JUST(VectorAt(inputs, 0)));  // input a
    if (broadcast_b) {
      ctx->b_index = ctx->SaveTensorForBackward(JUST(VectorAt(inputs, 1)));  // input b
    }
  }
  return Maybe<void>::Ok();
}

Maybe<void> BroadcastMatmul::Apply(const BroadcastMatmulCaptureState* ctx,
                                   const TensorTuple& out_grads, TensorTuple* in_grads) const {
  if (!ctx->requires_grad_a && !ctx->requires_grad_b) { return Maybe<void>::Ok(); }
  CHECK_EQ_OR_RETURN(out_grads.size(), 1) << "Out grad size should be equal to 1. ";
  in_grads->resize(2);
  const auto out_shape = JUST(VectorAt(out_grads, 0))->shape();
  const int64_t out_num_axes = out_shape->NumAxes();
  const size_t num_max_batch_dims = out_num_axes - 2;
  auto MakeGetBatchDim = [num_max_batch_dims](size_t num_dims, const Shape& shape_dim) {
    const int64_t num_batch_dims = num_dims - 2;
    const int64_t num_padding_dims = num_max_batch_dims - num_batch_dims;
    return [num_padding_dims, shape_dim](size_t index) {
      return index < num_padding_dims ? 1 : shape_dim.At(index - num_padding_dims);
    };
  };
  auto GetOutBatchDim = MakeGetBatchDim(out_num_axes, *out_shape);
  if (ctx->requires_grad_a) {
    std::shared_ptr<Tensor> broadcast_grad_a;
    const auto& input_b = ctx->SavedTensors().at(ctx->b_index);
    if (ctx->transpose_a) {
      broadcast_grad_a = JUST(functional::MatMul(input_b, JUST(VectorAt(out_grads, 0)),
                                                 ctx->transpose_b, true, ctx->alpha));
    } else {
      broadcast_grad_a = JUST(functional::MatMul(JUST(VectorAt(out_grads, 0)), input_b, false,
                                                 !(ctx->transpose_b), ctx->alpha));
    }
    if (ctx->broadcast_a) {
      const auto& input_a = JUST(VectorAt(ctx->SavedTensors(), ctx->a_index));
      const auto a_shape = input_a->shape();
      const int64_t a_num_axes = a_shape->NumAxes();

      std::vector<int32_t> a_reduce_vec;
      auto GetABatchDim = MakeGetBatchDim(a_num_axes, *a_shape);
      const int64_t a_out_num_dim_differ = out_num_axes - a_num_axes;
      for (int32_t i = 0; i < out_num_axes - 2; i++) {
        if (GetOutBatchDim(i) > GetABatchDim(i)
            || (GetOutBatchDim(i) == 1 && i < a_out_num_dim_differ)) {
          a_reduce_vec.push_back(i);
        }
      }
      JUST(VectorAt(*in_grads, 0)) =
          JUST(functional::ReduceSumLike(broadcast_grad_a, input_a, a_reduce_vec));
    } else {
      JUST(VectorAt(*in_grads, 0)) = broadcast_grad_a;
    }
  }

  if (ctx->requires_grad_b) {
    const auto& input_a = ctx->SavedTensors().at(ctx->a_index);
    if (ctx->b_num_axes == 2 && !ctx->transpose_a) {
      if (ctx->transpose_b) {
        JUST(VectorAt(*in_grads, 1)) = JUST(
            functional::BroadcastMatmulGradB(JUST(VectorAt(out_grads, 0)), input_a, ctx->alpha));
      } else {
        JUST(VectorAt(*in_grads, 1)) = JUST(
            functional::BroadcastMatmulGradB(input_a, JUST(VectorAt(out_grads, 0)), ctx->alpha));
      }
    } else {
      std::shared_ptr<Tensor> broadcast_grad_b;
      if (ctx->transpose_b) {
        broadcast_grad_b = JUST(functional::MatMul(JUST(VectorAt(out_grads, 0)), input_a, true,
                                                   ctx->transpose_a, ctx->alpha));
      } else {
        broadcast_grad_b = JUST(functional::MatMul(input_a, JUST(VectorAt(out_grads, 0)),
                                                   !ctx->transpose_a, false, ctx->alpha));
      }
      if (ctx->broadcast_b) {
        const auto& input_b = JUST(VectorAt(ctx->SavedTensors(), ctx->b_index));
        const auto b_shape = input_b->shape();
        std::vector<int32_t> b_reduce_vec;
        auto GetBBatchDim = MakeGetBatchDim(ctx->b_num_axes, *b_shape);
        const int64_t b_out_num_dim_differ = out_num_axes - ctx->b_num_axes;
        for (int32_t i = 0; i < out_num_axes - 2; i++) {
          if (GetOutBatchDim(i) > GetBBatchDim(i)
              || (GetOutBatchDim(i) == 1 && i < b_out_num_dim_differ)) {
            b_reduce_vec.push_back(i);
          }
        }
        JUST(VectorAt(*in_grads, 1)) =
            JUST(functional::ReduceSumLike(broadcast_grad_b, input_b, b_reduce_vec));
      } else {
        JUST(VectorAt(*in_grads, 1)) = broadcast_grad_b;
      }
    }
  }
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("matmul", Matmul);
REGISTER_OP_EXPR_GRAD_FUNCTION("batch_matmul", Matmul);
REGISTER_OP_EXPR_GRAD_FUNCTION("broadcast_matmul", BroadcastMatmul);

}  // namespace one
}  // namespace oneflow
