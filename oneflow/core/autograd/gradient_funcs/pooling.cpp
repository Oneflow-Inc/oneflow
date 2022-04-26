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

struct PoolingCaptureState : public AutoGradCaptureState {
  bool requires_grad;
  size_t input_index;
  size_t output_index;
  size_t indice_index;

  std::string data_format;
  std::vector<int32_t> padding;
  std::vector<int32_t> kernel_size;
  std::vector<int32_t> stride;
  std::vector<int32_t> dilation;
  bool return_indices;
  bool ceil_mode;
};

class PoolingNdGrad : public OpExprGradFunction<PoolingCaptureState> {
 public:
  virtual ~PoolingNdGrad() = default;

  using OpExprGradFunction<PoolingCaptureState>::Init;

  Maybe<void> Init(const OpExpr& op, const std::string& mode);
  Maybe<void> Capture(PoolingCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const PoolingCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  std::string mode_;
  AttrMap base_attrs_;
};

Maybe<void> PoolingNdGrad::Init(const OpExpr& op, const std::string& mode) {
  const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  mode_ = mode;
  return Maybe<void>::Ok();
}

Maybe<void> PoolingNdGrad::Capture(PoolingCaptureState* ctx, const TensorTuple& inputs,
                                   const TensorTuple& outputs, const AttrMap& attrs) const {
  ctx->requires_grad = inputs.at(0)->requires_grad();
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }

  ctx->input_index = ctx->SaveTensorForBackward(inputs.at(0));
  ctx->output_index = ctx->SaveTensorForBackward(outputs.at(0));
  ctx->indice_index = ctx->SaveTensorForBackward(outputs.at(1));

  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->data_format = JUST(composed_attrs.GetAttr<std::string>("data_format"));
  ctx->padding = JUST(composed_attrs.GetAttr<std::vector<int32_t>>("padding"));
  ctx->kernel_size = JUST(composed_attrs.GetAttr<std::vector<int32_t>>("kernel_size"));
  ctx->stride = JUST(composed_attrs.GetAttr<std::vector<int32_t>>("stride"));
  ctx->dilation = JUST(composed_attrs.GetAttr<std::vector<int32_t>>("dilation"));
  ctx->return_indices = JUST(composed_attrs.GetAttr<bool>("return_indices"));
  ctx->ceil_mode = JUST(composed_attrs.GetAttr<bool>("ceil_mode"));
  return Maybe<void>::Ok();
}

Maybe<void> PoolingNdGrad::Apply(const PoolingCaptureState* ctx, const TensorTuple& out_grads,
                                 TensorTuple* in_grads) const {
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
  CHECK_LE_OR_RETURN(out_grads.size(), 2);

  int32_t ndims = ctx->kernel_size.size();
  const auto& input = ctx->SavedTensors().at(ctx->input_index);
  const auto& output = ctx->SavedTensors().at(ctx->output_index);
  const auto& indice = ctx->SavedTensors().at(ctx->indice_index);

  in_grads->resize(1);
  in_grads->at(0) = JUST(functional::PoolingNdGrad(
      input, output, indice, out_grads.at(0), mode_, ndims, ctx->data_format, ctx->padding,
      ctx->kernel_size, ctx->stride, ctx->dilation, ctx->return_indices, ctx->ceil_mode));

  return Maybe<void>::Ok();
}

}  // namespace

class MaxpoolNdGrad final : public PoolingNdGrad {
 public:
  Maybe<void> Init(const OpExpr& op) override { return PoolingNdGrad::Init(op, "max"); }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("maxpool_1d", MaxpoolNdGrad);
REGISTER_OP_EXPR_GRAD_FUNCTION("maxpool_2d", MaxpoolNdGrad);
REGISTER_OP_EXPR_GRAD_FUNCTION("maxpool_3d", MaxpoolNdGrad);

}  // namespace one
}  // namespace oneflow
