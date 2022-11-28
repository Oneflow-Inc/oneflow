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
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

namespace {

struct TFPoolCaptureState : public AutoGradCaptureState {
  bool requires_grad = false;
  size_t input_index = 0;
  size_t output_index = 0;

  std::string data_format;
  std::string padding;
  std::vector<int32_t> padding_before;
  std::vector<int32_t> padding_after;
  std::vector<int32_t> pool_size;
  std::vector<int32_t> strides;
  bool ceil_mode = false;
};

class TFPoolNdGrad : public OpExprGradFunction<TFPoolCaptureState> {
 public:
  virtual ~TFPoolNdGrad() = default;

  using OpExprGradFunction<TFPoolCaptureState>::Init;

  Maybe<void> Init(const OpExpr& op, const std::string& mode);
  Maybe<void> Capture(TFPoolCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const TFPoolCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  std::string mode_;
  AttrMap base_attrs_;
};

Maybe<void> TFPoolNdGrad::Init(const OpExpr& op, const std::string& mode) {
  const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  mode_ = mode;
  return Maybe<void>::Ok();
}

Maybe<void> TFPoolNdGrad::Capture(TFPoolCaptureState* ctx, const TensorTuple& inputs,
                                  const TensorTuple& outputs, const AttrMap& attrs) const {
  ctx->requires_grad = inputs.at(0)->requires_grad();
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }

  ctx->input_index = ctx->SaveTensorForBackward(inputs.at(0));
  ctx->output_index = ctx->SaveTensorForBackward(outputs.at(0));

  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->data_format = JUST(composed_attrs.GetAttr<std::string>("data_format"));
  ctx->padding = JUST(composed_attrs.GetAttr<std::string>("padding"));
  ctx->padding_before = JUST(composed_attrs.GetAttr<std::vector<int32_t>>("padding_before"));
  ctx->padding_after = JUST(composed_attrs.GetAttr<std::vector<int32_t>>("padding_after"));
  ctx->pool_size = JUST(composed_attrs.GetAttr<std::vector<int32_t>>("pool_size"));
  ctx->strides = JUST(composed_attrs.GetAttr<std::vector<int32_t>>("strides"));
  ctx->ceil_mode = JUST(composed_attrs.GetAttr<bool>("ceil_mode"));
  return Maybe<void>::Ok();
}

Maybe<void> TFPoolNdGrad::Apply(const TFPoolCaptureState* ctx, const TensorTuple& out_grads,
                                TensorTuple* in_grads) const {
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)

  int32_t ndims = ctx->pool_size.size();
  const auto& input = ctx->SavedTensors().at(ctx->input_index);
  const auto& output = ctx->SavedTensors().at(ctx->output_index);

  in_grads->resize(1);
  (*in_grads)[0] = JUST(functional::TFPoolNdGrad(
      input, output, out_grads[0], mode_, ndims, ctx->data_format, ctx->padding,
      ctx->padding_before, ctx->padding_after, ctx->pool_size, ctx->strides, ctx->ceil_mode));

  return Maybe<void>::Ok();
}

}  // namespace

class TFMaxPoolNdGrad final : public TFPoolNdGrad {
 public:
  Maybe<void> Init(const OpExpr& op) override { return TFPoolNdGrad::Init(op, "tf_max"); }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("tf_max_pool_1d", TFMaxPoolNdGrad);
REGISTER_OP_EXPR_GRAD_FUNCTION("tf_max_pool_2d", TFMaxPoolNdGrad);
REGISTER_OP_EXPR_GRAD_FUNCTION("tf_max_pool_3d", TFMaxPoolNdGrad);

class TFAvgPoolNdGrad final : public TFPoolNdGrad {
 public:
  Maybe<void> Init(const OpExpr& op) override { return TFPoolNdGrad::Init(op, "tf_avg"); }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("tf_avg_pool_1d", TFAvgPoolNdGrad);
REGISTER_OP_EXPR_GRAD_FUNCTION("tf_avg_pool_2d", TFAvgPoolNdGrad);
REGISTER_OP_EXPR_GRAD_FUNCTION("tf_avg_pool_3d", TFAvgPoolNdGrad);

}  // namespace one
}  // namespace oneflow
