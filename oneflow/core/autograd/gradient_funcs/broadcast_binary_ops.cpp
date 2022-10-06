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
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct BroadcastBinaryCaptureState : public AutoGradCaptureState {
  int x_index = -1;
  int y_index = -1;
  int z_index = -1;
  bool x_requires_grad = false;
  bool y_requires_grad = false;
  bool broadcast_x = false;
  bool broadcast_y = false;
};

class BroadcastBinaryGrad : public OpExprGradFunction<BroadcastBinaryCaptureState> {
 public:
  BroadcastBinaryGrad() = default;
  virtual ~BroadcastBinaryGrad() = default;

  virtual Maybe<void> Init(const OpExpr& op) override { return Maybe<void>::Ok(); }

  Maybe<void> Capture(BroadcastBinaryCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 2);   // NOLINT(maybe-need-error-msg)
    CHECK_EQ_OR_RETURN(outputs.size(), 1);  // NOLINT(maybe-need-error-msg)
    ctx->x_requires_grad = inputs.at(0)->requires_grad();
    ctx->y_requires_grad = inputs.at(1)->requires_grad();
    ctx->broadcast_x = (*inputs.at(0)->shape() != *outputs.at(0)->shape());
    ctx->broadcast_y = (*inputs.at(1)->shape() != *outputs.at(0)->shape());
    return SaveTensorForBackward(ctx, inputs, outputs);
  }

 protected:
  virtual Maybe<void> SaveTensorForBackward(BroadcastBinaryCaptureState* ctx,
                                            const TensorTuple& inputs,
                                            const TensorTuple& outputs) const = 0;
};

class BroadcastAdd : public BroadcastBinaryGrad {
 public:
  Maybe<void> Apply(const BroadcastBinaryCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    in_grads->resize(2);
    if (ctx->x_requires_grad) {
      if (ctx->broadcast_x) {
        const auto& x = ctx->SavedTensors().at(ctx->x_index);
        in_grads->at(0) = JUST(functional::BroadcastReduceSumLike(out_grads.at(0), x));
      } else {
        in_grads->at(0) = out_grads.at(0);
      }
    }
    if (ctx->y_requires_grad) {
      if (ctx->broadcast_y) {
        const auto& y = ctx->SavedTensors().at(ctx->y_index);
        in_grads->at(1) = JUST(functional::BroadcastReduceSumLike(out_grads.at(0), y));
      } else {
        in_grads->at(1) = out_grads.at(0);
      }
    }
    return Maybe<void>::Ok();
  }

 protected:
  Maybe<void> SaveTensorForBackward(BroadcastBinaryCaptureState* ctx, const TensorTuple& inputs,
                                    const TensorTuple& outputs) const override {
    if (ctx->x_requires_grad && ctx->broadcast_x) {
      ctx->x_index = ctx->SaveTensorForBackward(inputs.at(0));
    }
    if (ctx->y_requires_grad && ctx->broadcast_y) {
      ctx->y_index = ctx->SaveTensorForBackward(inputs.at(1));
    }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("broadcast_add", BroadcastAdd);

class BroadcastSub : public BroadcastBinaryGrad {
 public:
  Maybe<void> Apply(const BroadcastBinaryCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    in_grads->resize(2);
    if (ctx->x_requires_grad) {
      if (ctx->broadcast_x) {
        const auto& x = ctx->SavedTensors().at(ctx->x_index);
        in_grads->at(0) = JUST(functional::BroadcastReduceSumLike(out_grads.at(0), x));
      } else {
        in_grads->at(0) = out_grads.at(0);
      }
    }
    if (ctx->y_requires_grad) {
      const auto& grad = JUST(functional::ScalarMul(out_grads.at(0), Scalar(-1.f), false));
      if (ctx->broadcast_y) {
        const auto& y = ctx->SavedTensors().at(ctx->y_index);
        in_grads->at(1) = JUST(functional::BroadcastReduceSumLike(grad, y));
      } else {
        in_grads->at(1) = grad;
      }
    }
    return Maybe<void>::Ok();
  }

 protected:
  Maybe<void> SaveTensorForBackward(BroadcastBinaryCaptureState* ctx, const TensorTuple& inputs,
                                    const TensorTuple& outputs) const override {
    if (ctx->x_requires_grad && ctx->broadcast_x) {
      ctx->x_index = ctx->SaveTensorForBackward(inputs.at(0));
    }
    if (ctx->y_requires_grad && ctx->broadcast_y) {
      ctx->y_index = ctx->SaveTensorForBackward(inputs.at(1));
    }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("broadcast_sub", BroadcastSub);

class BroadcastMul : public BroadcastBinaryGrad {
 public:
  Maybe<void> Apply(const BroadcastBinaryCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    in_grads->resize(2);
    if (ctx->x_requires_grad) {
      const auto& y = ctx->SavedTensors().at(ctx->y_index);
      const auto& x_grad = JUST(functional::Mul(out_grads.at(0), y));
      if (ctx->broadcast_x) {
        const auto& x = ctx->SavedTensors().at(ctx->x_index);
        in_grads->at(0) = JUST(functional::BroadcastReduceSumLike(x_grad, x));
      } else {
        in_grads->at(0) = x_grad;
      }
    }
    if (ctx->y_requires_grad) {
      const auto& x = ctx->SavedTensors().at(ctx->x_index);
      const auto& y_grad = JUST(functional::Mul(out_grads.at(0), x));
      if (ctx->broadcast_y) {
        const auto& y = ctx->SavedTensors().at(ctx->y_index);
        in_grads->at(1) = JUST(functional::BroadcastReduceSumLike(y_grad, y));
      } else {
        in_grads->at(1) = y_grad;
      }
    }
    return Maybe<void>::Ok();
  }

 protected:
  Maybe<void> SaveTensorForBackward(BroadcastBinaryCaptureState* ctx, const TensorTuple& inputs,
                                    const TensorTuple& outputs) const override {
    if (ctx->x_requires_grad) {
      ctx->y_index = ctx->SaveTensorForBackward(inputs.at(1));
      if (ctx->broadcast_x) { ctx->x_index = ctx->SaveTensorForBackward(inputs.at(0)); }
    }
    if (ctx->y_requires_grad) {
      if (ctx->x_index == -1 /*x has not been saved*/) {
        ctx->x_index = ctx->SaveTensorForBackward(inputs.at(0));
      }
      if (ctx->broadcast_y && ctx->y_index == -1 /*y has not been saved*/) {
        ctx->y_index = ctx->SaveTensorForBackward(inputs.at(1));
      }
    }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("broadcast_mul", BroadcastMul);

class BroadcastDiv : public BroadcastBinaryGrad {
 public:
  Maybe<void> Apply(const BroadcastBinaryCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    in_grads->resize(2);
    if (ctx->x_requires_grad) {
      const auto& y = ctx->SavedTensors().at(ctx->y_index);
      const auto& x_grad = JUST(functional::Div(out_grads.at(0), y));
      if (ctx->broadcast_x) {
        const auto& x = ctx->SavedTensors().at(ctx->x_index);
        in_grads->at(0) = JUST(functional::BroadcastReduceSumLike(x_grad, x));
      } else {
        in_grads->at(0) = x_grad;
      }
    }
    if (ctx->y_requires_grad) {
      const auto& y = ctx->SavedTensors().at(ctx->y_index);
      const auto& z = ctx->SavedTensors().at(ctx->z_index);
      in_grads->at(1) = JUST(functional::DivGrad(out_grads.at(0), z, y));
    }
    return Maybe<void>::Ok();
  }

 protected:
  Maybe<void> SaveTensorForBackward(BroadcastBinaryCaptureState* ctx, const TensorTuple& inputs,
                                    const TensorTuple& outputs) const override {
    if (ctx->x_requires_grad) {
      ctx->y_index = ctx->SaveTensorForBackward(inputs.at(1));
      if (ctx->broadcast_x) { ctx->x_index = ctx->SaveTensorForBackward(inputs.at(0)); }
    }
    if (ctx->y_requires_grad) {
      if (ctx->y_index == -1 /*y has not been saved*/) {
        ctx->y_index = ctx->SaveTensorForBackward(inputs.at(1));
      }
      ctx->z_index = ctx->SaveTensorForBackward(outputs.at(0));
    }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("broadcast_div", BroadcastDiv);

class BroadcastPow : public BroadcastBinaryGrad {
 public:
  Maybe<void> Apply(const BroadcastBinaryCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    const auto& x = ctx->SavedTensors().at(ctx->x_index);
    const auto& y = ctx->SavedTensors().at(ctx->y_index);
    in_grads->resize(2);
    if (ctx->x_requires_grad) {
      (*in_grads)[0] = JUST(functional::BroadcastPowXGrad(x, y, out_grads[0]));
    }
    if (ctx->y_requires_grad) {
      (*in_grads)[1] = JUST(functional::BroadcastPowYGrad(x, y, out_grads[0]));
    }
    return Maybe<void>::Ok();
  }

 protected:
  Maybe<void> SaveTensorForBackward(BroadcastBinaryCaptureState* ctx, const TensorTuple& inputs,
                                    const TensorTuple& outputs) const override {
    ctx->x_index = ctx->SaveTensorForBackward(inputs[0]);
    ctx->y_index = ctx->SaveTensorForBackward(inputs[1]);
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("broadcast_pow", BroadcastPow);

class BroadcastMinMax : public BroadcastBinaryGrad {
 public:
  Maybe<void> Apply(const BroadcastBinaryCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    const auto& out_shape = *(out_grads.at(0)->shape());
    in_grads->resize(2);
    if (ctx->x_requires_grad || ctx->y_requires_grad) {
      const auto& x = ctx->SavedTensors().at(ctx->x_index);
      const auto& y = ctx->SavedTensors().at(ctx->y_index);
      auto broad_x_ = x;
      auto broad_y_ = y;
      if (ctx->broadcast_x) {
        const auto& x_shape = *(x->shape());
        const Shape& left_extended_x_shape =
            CreateLeftExtendedShape(ShapeView(x_shape), out_shape.NumAxes());
        if (left_extended_x_shape == out_shape) {
          broad_x_ = JUST(functional::ReshapeLike(x, JUST(VectorAt(out_grads, 0))));
        } else {
          const AxisVector& broadcast_axis_vec = left_extended_x_shape.Axes4BroadcastTo(out_shape);
          const std::vector<int32_t> x_axis =
              std::vector<int32_t>{broadcast_axis_vec.begin(), broadcast_axis_vec.end()};
          broad_x_ = JUST(functional::BroadcastLike(x, JUST(VectorAt(out_grads, 0)), x_axis));
        }
      }
      if (ctx->broadcast_y) {
        const auto& y_shape = *(y->shape());
        const Shape& left_extended_y_shape =
            CreateLeftExtendedShape(ShapeView(y_shape), out_shape.NumAxes());
        if (left_extended_y_shape == out_shape) {
          broad_y_ = JUST(functional::ReshapeLike(y, JUST(VectorAt(out_grads, 0))));
        } else {
          const AxisVector& broadcast_axis_vec = left_extended_y_shape.Axes4BroadcastTo(out_shape);
          const std::vector<int32_t> y_axis =
              std::vector<int32_t>{broadcast_axis_vec.begin(), broadcast_axis_vec.end()};
          broad_y_ = JUST(functional::BroadcastLike(y, JUST(VectorAt(out_grads, 0)), y_axis));
        }
      }
      const auto& broad_grads =
          JUST(elementwise_grad_functor_(out_grads.at(0), broad_x_, broad_y_));
      if (ctx->x_requires_grad) {
        if (ctx->broadcast_x) {
          in_grads->at(0) = JUST(functional::BroadcastReduceSumLike(broad_grads->at(0), x));
        } else {
          in_grads->at(0) = broad_grads->at(0);
        }
      }
      if (ctx->y_requires_grad) {
        if (ctx->broadcast_y) {
          in_grads->at(1) = JUST(functional::BroadcastReduceSumLike(broad_grads->at(1), y));
        } else {
          in_grads->at(1) = broad_grads->at(1);
        }
      }
    }
    return Maybe<void>::Ok();
  }

 protected:
  Maybe<void> SaveTensorForBackward(BroadcastBinaryCaptureState* ctx, const TensorTuple& inputs,
                                    const TensorTuple& outputs) const override {
    if (ctx->x_requires_grad || ctx->y_requires_grad) {
      ctx->x_index = ctx->SaveTensorForBackward(inputs.at(0));
      ctx->y_index = ctx->SaveTensorForBackward(inputs.at(1));
    }
    return Maybe<void>::Ok();
  }

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

class BroadcastFMod : public BroadcastBinaryGrad {
 public:
  Maybe<void> Apply(const BroadcastBinaryCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    const auto& out_shape = *(JUST(VectorAt(out_grads, 0))->shape());
    in_grads->resize(2);
    if (ctx->x_requires_grad || ctx->y_requires_grad) {
      const auto& x = JUST(VectorAt(ctx->SavedTensors(), ctx->x_index));
      const auto& y = JUST(VectorAt(ctx->SavedTensors(), ctx->y_index));
      auto broad_x_ = x;
      auto broad_y_ = y;
      if (ctx->broadcast_x) {
        const auto& x_shape = *(x->shape());
        const Shape& left_extended_x_shape =
            CreateLeftExtendedShape(ShapeView(x_shape), out_shape.NumAxes());
        if (left_extended_x_shape == out_shape) {
          broad_x_ = JUST(functional::ReshapeLike(x, JUST(VectorAt(out_grads, 0))));
        } else {
          const AxisVector& broadcast_axis_vec = left_extended_x_shape.Axes4BroadcastTo(out_shape);
          const std::vector<int32_t> x_axis =
              std::vector<int32_t>{broadcast_axis_vec.begin(), broadcast_axis_vec.end()};
          broad_x_ = JUST(functional::BroadcastLike(x, JUST(VectorAt(out_grads, 0)), x_axis));
        }
      }
      if (ctx->broadcast_y) {
        const auto& y_shape = *(y->shape());
        const Shape& left_extended_y_shape =
            CreateLeftExtendedShape(ShapeView(y_shape), out_shape.NumAxes());
        if (left_extended_y_shape == out_shape) {
          broad_y_ = JUST(functional::ReshapeLike(y, JUST(VectorAt(out_grads, 0))));
        } else {
          const AxisVector& broadcast_axis_vec = left_extended_y_shape.Axes4BroadcastTo(out_shape);
          const std::vector<int32_t> y_axis =
              std::vector<int32_t>{broadcast_axis_vec.begin(), broadcast_axis_vec.end()};
          broad_y_ = JUST(functional::BroadcastLike(y, JUST(VectorAt(out_grads, 0)), y_axis));
        }
      }
      if (ctx->x_requires_grad) {
        if (ctx->broadcast_x) {
          JUST(VectorAt(*in_grads, 0)) =
              JUST(functional::BroadcastReduceSumLike(JUST(VectorAt(out_grads, 0)), x));
        } else {
          JUST(VectorAt(*in_grads, 0)) = JUST(VectorAt(out_grads, 0));
        }
      }
      if (ctx->y_requires_grad) {
        auto result = JUST(functional::TruncDiv(broad_x_, broad_y_));
        result = JUST(functional::Mul(JUST(VectorAt(out_grads, 0)), result));
        JUST(functional::ScalarMul(result, Scalar(-1.f), true));
        if (ctx->broadcast_y) {
          in_grads->at(1) = JUST(functional::BroadcastReduceSumLike(result, y));
        } else {
          in_grads->at(1) = result;
        }
      }
    }
    return Maybe<void>::Ok();
  }

 protected:
  Maybe<void> SaveTensorForBackward(BroadcastBinaryCaptureState* ctx, const TensorTuple& inputs,
                                    const TensorTuple& outputs) const override {
    if (ctx->x_requires_grad && ctx->broadcast_x) {
      ctx->x_index = ctx->SaveTensorForBackward(JUST(VectorAt(inputs, 0)));
    }
    if (ctx->y_requires_grad) {
      ctx->x_index = ctx->SaveTensorForBackward(JUST(VectorAt(inputs, 0)));
      ctx->y_index = ctx->SaveTensorForBackward(JUST(VectorAt(inputs, 1)));
    }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("broadcast_fmod", BroadcastFMod);

}  // namespace one
}  // namespace oneflow
