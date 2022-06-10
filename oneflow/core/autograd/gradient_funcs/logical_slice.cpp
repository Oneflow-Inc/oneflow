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

struct LogicalSliceCaptureState : public AutoGradCaptureState {
  Shape like_shape;
  std::vector<int64_t> start;
  std::vector<int64_t> stop;
  std::vector<int64_t> step;
  Symbol<NdSbp> in_sbp;
};

class LogicalSlice : public OpExprGradFunction<LogicalSliceCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr) << "LogicalSlice op_expr is null";
    base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(LogicalSliceCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1) << "LogicalSlice input size must be 1";
    CHECK_EQ_OR_RETURN(outputs.size(), 1) << "LogicalSlice output size must be 1";

    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    ctx->start = JUST(composed_attrs.GetAttr<std::vector<int64_t>>("start"));
    ctx->stop = JUST(composed_attrs.GetAttr<std::vector<int64_t>>("stop"));
    ctx->step = JUST(composed_attrs.GetAttr<std::vector<int64_t>>("step"));
    ctx->like_shape = *(inputs[0]->shape());
    ctx->in_sbp = JUST(inputs[0]->nd_sbp());
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const LogicalSliceCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    in_grads->resize(1);
    std::shared_ptr<Tensor> zeros;
    if (out_grads[0]->is_local()) {
      zeros = JUST(functional::Constant(ctx->like_shape, 0, out_grads[0]->dtype(),
                                        JUST(out_grads[0]->device())));
    } else {
      const auto& parallel_desc = JUST(out_grads[0]->parallel_desc());
      zeros = JUST(functional::ConsistentConstant(ctx->like_shape, 0, out_grads[0]->dtype(),
                                                  parallel_desc, *JUST(GetSbpList(ctx->in_sbp))));
    }
    (*in_grads)[0] =
        JUST(functional::LogicalSliceAssign(zeros, out_grads[0], ctx->start, ctx->stop, ctx->step));
    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

struct LogicalSliceAssignCaptureState : public AutoGradCaptureState {
  bool requires_grad_ref = false;
  bool requires_grad_value = false;
  std::vector<int64_t> start;
  std::vector<int64_t> stop;
  std::vector<int64_t> step;
  Shape value_shape;  // used to calculate ref gradient
  Symbol<NdSbp> value_sbp;
};

class LogicalSliceAssign : public OpExprGradFunction<LogicalSliceAssignCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr) << "LogicalSliceAssign op_expr is null";

    base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(LogicalSliceAssignCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 2) << "LogicalSliceAssign input size must be 2";
    CHECK_EQ_OR_RETURN(outputs.size(), 1) << "LogicalSliceAssign output size must be 1";
    ctx->requires_grad_ref = inputs[0]->requires_grad();
    ctx->requires_grad_value = inputs[1]->requires_grad();
    if (!ctx->requires_grad_ref && !ctx->requires_grad_value) { return Maybe<void>::Ok(); }

    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    ctx->start = JUST(composed_attrs.GetAttr<std::vector<int64_t>>("start"));
    ctx->stop = JUST(composed_attrs.GetAttr<std::vector<int64_t>>("stop"));
    ctx->step = JUST(composed_attrs.GetAttr<std::vector<int64_t>>("step"));

    if (ctx->requires_grad_ref) {
      ctx->value_shape = *(inputs[1]->shape());
      ctx->value_sbp = JUST(inputs[1]->nd_sbp());
    }
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const LogicalSliceAssignCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    in_grads->resize(2);

    if (ctx->requires_grad_ref) {
      std::shared_ptr<Tensor> zeros;
      if (out_grads[0]->is_local()) {
        zeros = JUST(functional::Constant(ctx->value_shape, 0, out_grads[0]->dtype(),
                                          JUST(out_grads[0]->device())));
      } else {
        const auto& parallel_desc = JUST(out_grads[0]->parallel_desc());
        zeros =
            JUST(functional::ConsistentConstant(ctx->value_shape, 0, out_grads[0]->dtype(),
                                                parallel_desc, *JUST(GetSbpList(ctx->value_sbp))));
      }
      (*in_grads)[0] = JUST(functional::LogicalSliceAssign(
          JUST(functional::Identity(out_grads[0])), zeros, ctx->start, ctx->stop, ctx->step));
    }
    if (ctx->requires_grad_value) {
      (*in_grads)[1] = JUST(functional::LogicalSlice(out_grads[0], ctx->start, ctx->stop, ctx->step,
                                                     /*enable_view_slice=*/false));
    }
    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("logical_slice_assign", LogicalSliceAssign);
REGISTER_OP_EXPR_GRAD_FUNCTION("logical_slice", LogicalSlice);

}  // namespace one
}  // namespace oneflow
