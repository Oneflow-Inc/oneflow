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

struct PadNdCaptureState : public AutoGradCaptureState {
  bool requires_grad = false;
  std::vector<int64_t> paddings{};
};

class PadNd : public OpExprGradFunction<PadNdCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const UserOpExpr* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
    base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(PadNdCaptureState* ctx, const TensorTuple& inputs, const TensorTuple& outputs,
                      const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);   // NOLINT(maybe-need-error-msg)
    CHECK_EQ_OR_RETURN(outputs.size(), 1);  // NOLINT(maybe-need-error-msg)
    ctx->requires_grad = JUST(VectorAt(inputs, 0))->requires_grad();
    if (!ctx->requires_grad) { return Maybe<void>::Ok(); }

    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    ctx->paddings = JUST(composed_attrs.GetAttr<std::vector<int64_t>>("padding"));
    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

class ReflectionPadNd : public PadNd {
 public:
  Maybe<void> Apply(const PadNdCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)
    in_grads->resize(1);
    if (ctx->requires_grad) {
      (*in_grads)[0] =
          JUST(functional::PadGrad(JUST(VectorAt(out_grads, 0)), ctx->paddings, "reflect", 0));
    }
    return Maybe<void>::Ok();
  }
};

class ReplicationPadNd : public PadNd {
 public:
  Maybe<void> Apply(const PadNdCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)
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
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
    base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(ConstantPadNdCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);   // NOLINT(maybe-need-error-msg)
    CHECK_EQ_OR_RETURN(outputs.size(), 1);  // NOLINT(maybe-need-error-msg)
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
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)
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
REGISTER_OP_EXPR_GRAD_FUNCTION("reflection_pad1d", ReflectionPadNd);
REGISTER_OP_EXPR_GRAD_FUNCTION("reflection_pad2d", ReflectionPadNd);
REGISTER_OP_EXPR_GRAD_FUNCTION("replication_pad1d", ReplicationPadNd);
REGISTER_OP_EXPR_GRAD_FUNCTION("replication_pad2d", ReplicationPadNd);

}  // namespace one
}  // namespace oneflow
