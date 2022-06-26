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
#include "oneflow/core/common/container_util.h"

namespace oneflow {
namespace one {

struct Pad2dCaptureState : public AutoGradCaptureState {
  bool requires_grad;
  std::vector<int64_t> paddings;
};

class Pad2d : public OpExprGradFunction<Pad2dCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const UserOpExpr* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);
    base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(Pad2dCaptureState* ctx, const TensorTuple& inputs, const TensorTuple& outputs,
                      const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);
    CHECK_EQ_OR_RETURN(outputs.size(), 1);
    ctx->requires_grad = JUST(VectorAt(inputs, 0))->requires_grad();
    if (!ctx->requires_grad) { return Maybe<void>::Ok(); }

    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    ctx->paddings = JUST(composed_attrs.GetAttr<std::vector<int64_t>>("padding"));
    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

class ReflectionPad2d : public Pad2d {
 public:
  Maybe<void> Apply(const Pad2dCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    in_grads->resize(1);
    if (ctx->requires_grad) {
      (*in_grads)[0] =
          JUST(functional::PadGrad(JUST(VectorAt(out_grads, 0)), ctx->paddings, "reflect", 0));
    }
    return Maybe<void>::Ok();
  }
};

class ReplicationPad2d : public Pad2d {
 public:
  Maybe<void> Apply(const Pad2dCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    in_grads->resize(1);
    if (ctx->requires_grad) {
      (*in_grads)[0] =
          JUST(functional::PadGrad(JUST(VectorAt(out_grads, 0)), ctx->paddings, "replicate", 0));
    }
    return Maybe<void>::Ok();
  }
};

struct ConstantPadNdCaptureState : public AutoGradCaptureState {
  bool requires_grad;
  std::vector<int64_t> paddings;
};

class ConstantPadNd : public OpExprGradFunction<ConstantPadNdCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const UserOpExpr* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);
    base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(ConstantPadNdCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);
    CHECK_EQ_OR_RETURN(outputs.size(), 1);
    const std::shared_ptr<Tensor>& input_0 = JUST(VectorAt(inputs, 0));
    ctx->requires_grad = input_0->requires_grad();
    if (!ctx->requires_grad) { return Maybe<void>::Ok(); }

    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    ctx->paddings = JUST(composed_attrs.GetAttr<std::vector<int64_t>>("padding"));
    for (int i = 0; i < ctx->paddings.size(); i++) { ctx->paddings[i] = -ctx->paddings[i]; }
    return Maybe<void>::Ok();
  }
  Maybe<void> Apply(const ConstantPadNdCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    in_grads->resize(1);
    if (ctx->requires_grad) {
      (*in_grads)[0] =
          JUST(functional::Pad(JUST(VectorAt(out_grads, 0)), ctx->paddings, "constant", Scalar(0)));
    }
    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("pad", ConstantPadNd);
REGISTER_OP_EXPR_GRAD_FUNCTION("reflection_pad2d", ReflectionPad2d);
REGISTER_OP_EXPR_GRAD_FUNCTION("replication_pad2d", ReplicationPad2d);

}  // namespace one
}  // namespace oneflow
