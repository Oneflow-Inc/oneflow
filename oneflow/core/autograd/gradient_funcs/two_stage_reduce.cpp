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
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

enum class ReduceMode : int32_t {
  kMin = 0,
  kMax = 1,
};

struct ReduceDeviceCaptureState : public AutoGradCaptureState {
  std::vector<int32_t> axis;
  bool requires_grad = false;
  size_t mask_index = -1;
  size_t count_index = -1;
};

template<ReduceMode mode>
class ReduceDevice : public OpExprGradFunction<ReduceDeviceCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(op_expr);  // NOLINT(maybe-need-error-msg)
    base_attrs_ = MakeAttrMapFromUserOpConf(op_expr->proto());
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(ReduceDeviceCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);  // NOLINT(maybe-need-error-msg)
    ctx->requires_grad = inputs.at(0)->requires_grad();
    if (!ctx->requires_grad) { return Maybe<void>::Ok(); }

    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    ctx->axis = JUST(composed_attrs.GetAttr<std::vector<int32_t>>("axis"));
    ctx->mask_index = ctx->SaveTensorForBackward(outputs.at(1));   // mask
    ctx->count_index = ctx->SaveTensorForBackward(outputs.at(2));  // count
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const ReduceDeviceCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    if (!ctx->requires_grad) { return Maybe<void>::Ok(); }

    CHECK_EQ_OR_RETURN(out_grads.size(), 3);  // NOLINT(maybe-need-error-msg)
    const auto& mask = ctx->SavedTensors().at(ctx->mask_index);
    const auto& count = ctx->SavedTensors().at(ctx->count_index);
    in_grads->resize(1);
    if (mode == ReduceMode::kMin) {
      in_grads->at(0) =
          JUST(functional::ReduceMinDeviceStageGrad(out_grads.at(0), mask, count, ctx->axis));
    } else {
      in_grads->at(0) =
          JUST(functional::ReduceMaxDeviceStageGrad(out_grads.at(0), mask, count, ctx->axis));
    }
    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("reduce_min_device_stage", ReduceDevice<ReduceMode::kMin>);
REGISTER_OP_EXPR_GRAD_FUNCTION("reduce_max_device_stage", ReduceDevice<ReduceMode::kMax>);

struct ReduceGlobalCaptureState : public AutoGradCaptureState {
  std::vector<int32_t> axis;
  bool requires_grad = false;
  bool keepdims = false;
  size_t mask_index = -1;
  size_t device_count_index = -1;
};

template<ReduceMode mode>
class ReduceGlobal : public OpExprGradFunction<ReduceGlobalCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
    base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(ReduceGlobalCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 2);   // NOLINT(maybe-need-error-msg)
    CHECK_EQ_OR_RETURN(outputs.size(), 2);  // NOLINT(maybe-need-error-msg)
    ctx->requires_grad = inputs.at(0)->requires_grad();
    if (!ctx->requires_grad) { return Maybe<void>::Ok(); }

    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    ctx->axis = JUST(composed_attrs.GetAttr<std::vector<int32_t>>("axis"));
    ctx->keepdims = JUST(composed_attrs.GetAttr<bool>("keepdims"));
    ctx->mask_index = ctx->SaveTensorForBackward(outputs.at(1));         // mask
    ctx->device_count_index = ctx->SaveTensorForBackward(inputs.at(1));  // device_count
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const ReduceGlobalCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
    CHECK_EQ_OR_RETURN(out_grads.size(), 2);  // NOLINT(maybe-need-error-msg)
    const auto& mask = ctx->SavedTensors().at(ctx->mask_index);
    const auto& device_count = ctx->SavedTensors().at(ctx->device_count_index);
    in_grads->resize(2);
    if (mode == ReduceMode::kMin) {
      in_grads->at(0) = JUST(functional::ReduceMinGlobalStageGrad(
          out_grads.at(0), mask, device_count, ctx->axis, ctx->keepdims));
    } else {
      in_grads->at(0) = JUST(functional::ReduceMaxGlobalStageGrad(
          out_grads.at(0), mask, device_count, ctx->axis, ctx->keepdims));
    }
    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("reduce_min_global_stage", ReduceGlobal<ReduceMode::kMin>);
REGISTER_OP_EXPR_GRAD_FUNCTION("reduce_max_global_stage", ReduceGlobal<ReduceMode::kMax>);

}  // namespace one
}  // namespace oneflow
