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
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

class BroadcastBinaryGrad : public OpExprGradFunction<AutoGradCaptureState> {
 public:
  BroadcastBinaryGrad() = default;
  virtual ~BroadcastBinaryGrad() = default;

  virtual Maybe<void> Init(const OpExpr& op) override { return Maybe<void>::Ok(); }

  Maybe<void> Capture(AutoGradCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 2);
    CHECK_EQ_OR_RETURN(outputs.size(), 1);
    ctx->SaveTensorForBackward(inputs.at(0));
    ctx->SaveTensorForBackward(inputs.at(1));
    ctx->SaveTensorForBackward(outputs.at(0));
    return Maybe<void>::Ok();
  }
};

class BroadcastAdd : public BroadcastBinaryGrad {
 public:
  Maybe<void> Apply(const AutoGradCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    const auto& x = ctx->SavedTensors().at(0);
    const auto& y = ctx->SavedTensors().at(1);
    in_grads->resize(2);
    if (x->requires_grad()) {
      in_grads->at(0) = JUST(functional::BroadcastReduceSumLike(out_grads.at(0), x));
    }
    if (y->requires_grad()) {
      in_grads->at(1) = JUST(functional::BroadcastReduceSumLike(out_grads.at(0), y));
    }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("broadcast_add", BroadcastAdd);

class BroadcastSub : public BroadcastBinaryGrad {
 public:
  Maybe<void> Apply(const AutoGradCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    const auto& x = ctx->SavedTensors().at(0);
    const auto& y = ctx->SavedTensors().at(1);
    in_grads->resize(2);
    if (x->requires_grad()) {
      in_grads->at(0) = JUST(functional::BroadcastReduceSumLike(out_grads.at(0), x));
    }
    if (y->requires_grad()) {
      const auto& grad = JUST(functional::ScalarMul(out_grads.at(0), Scalar(-1.f), false));
      in_grads->at(1) = JUST(functional::BroadcastReduceSumLike(grad, y));
    }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("broadcast_sub", BroadcastSub);

class BroadcastMul : public BroadcastBinaryGrad {
 public:
  Maybe<void> Apply(const AutoGradCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    const auto& x = ctx->SavedTensors().at(0);
    const auto& y = ctx->SavedTensors().at(1);
    in_grads->resize(2);
    if (x->requires_grad()) {
      const auto& x_grad = JUST(functional::Mul(out_grads.at(0), y));
      in_grads->at(0) = JUST(functional::BroadcastReduceSumLike(x_grad, x));
    }
    if (y->requires_grad()) {
      const auto& y_grad = JUST(functional::Mul(out_grads.at(0), x));
      in_grads->at(1) = JUST(functional::BroadcastReduceSumLike(y_grad, y));
    }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("broadcast_mul", BroadcastMul);

class BroadcastDiv : public BroadcastBinaryGrad {
 public:
  Maybe<void> Apply(const AutoGradCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    const auto& x = ctx->SavedTensors().at(0);
    const auto& y = ctx->SavedTensors().at(1);
    const auto& z = ctx->SavedTensors().at(2);
    in_grads->resize(2);
    if (x->requires_grad()) {
      const auto& x_grad = JUST(functional::Div(out_grads.at(0), y));
      in_grads->at(0) = JUST(functional::BroadcastReduceSumLike(x_grad, x));
    }
    if (y->requires_grad()) { in_grads->at(1) = JUST(functional::DivGrad(out_grads.at(0), z, y)); }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("broadcast_div", BroadcastDiv);

class BroadcastPow : public BroadcastBinaryGrad {
 public:
  Maybe<void> Apply(const AutoGradCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    const auto& x = ctx->SavedTensors().at(0);
    const auto& y = ctx->SavedTensors().at(1);
    const auto& z = ctx->SavedTensors().at(2);
    in_grads->resize(2);
    if (x->requires_grad()) {
      in_grads->at(0) = JUST(functional::BroadcastPowXGrad(out_grads.at(0), x, y, z));
    }
    if (y->requires_grad()) {
      in_grads->at(1) = JUST(functional::BroadcastPowYGrad(out_grads.at(0), x, y, z));
    }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("broadcast_pow", BroadcastPow);

class BroadcastMinMax : public BroadcastBinaryGrad {
 public:
  Maybe<void> Apply(const AutoGradCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    const auto& x = ctx->SavedTensors().at(0);
    const auto& y = ctx->SavedTensors().at(1);
    const auto& out = ctx->SavedTensors().at(2);
    const auto& out_shape = *(out->shape());
    in_grads->resize(2);
    if (x->requires_grad() || y->requires_grad()) {
      const auto& x_shape = *(x->shape());
      const auto& y_shape = *(y->shape());
      auto broad_x_ = x;
      auto broad_y_ = y;
      if (x_shape != out_shape) {
        const Shape& left_extended_x_shape =
            CreateLeftExtendedShape(ShapeView(x_shape), out_shape.NumAxes());
        const AxisVector& broadcast_axis_vec = left_extended_x_shape.Axes4BroadcastTo(out_shape);
        const std::vector<int32_t> x_axis =
            std::vector<int32_t>{broadcast_axis_vec.begin(), broadcast_axis_vec.end()};
        broad_x_ = JUST(functional::BroadcastLike(x, out, x_axis));
      }
      if (y_shape != out_shape) {
        const Shape& left_extended_y_shape =
            CreateLeftExtendedShape(ShapeView(y_shape), out_shape.NumAxes());
        const AxisVector& broadcast_axis_vec = left_extended_y_shape.Axes4BroadcastTo(out_shape);
        const std::vector<int32_t> y_axis =
            std::vector<int32_t>{broadcast_axis_vec.begin(), broadcast_axis_vec.end()};
        broad_y_ = JUST(functional::BroadcastLike(y, out, y_axis));
      }
      const auto& broad_grads =
          JUST(elementwise_grad_functor_(out_grads.at(0), broad_x_, broad_y_));
      if (x->requires_grad()) {
        in_grads->at(0) = JUST(functional::BroadcastReduceSumLike(broad_grads->at(0), x));
      }
      if (y->requires_grad()) {
        in_grads->at(1) = JUST(functional::BroadcastReduceSumLike(broad_grads->at(1), y));
      }
    }
    return Maybe<void>::Ok();
  }

 protected:
  std::function<Maybe<TensorTuple>(const std::shared_ptr<Tensor>&, const std::shared_ptr<Tensor>&,
                                   const std::shared_ptr<Tensor>&)>
      elementwise_grad_functor_;
};

class BroadcastMinimum : public BroadcastMinMax {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    JUST(BroadcastMinMax::Init(op));
    elementwise_grad_functor_ = functional::ElementwiseMinGrad;
    return Maybe<void>::Ok();
  }
};

class BroadcastMaximum : public BroadcastMinMax {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    JUST(BroadcastMinMax::Init(op));
    elementwise_grad_functor_ = functional::ElementwiseMaxGrad;
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("broadcast_minimum", BroadcastMinimum);
REGISTER_OP_EXPR_GRAD_FUNCTION("broadcast_maximum", BroadcastMaximum);

}  // namespace one
}  // namespace oneflow
