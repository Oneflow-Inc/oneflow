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

#include "oneflow/core/common/maybe.h"
#include "oneflow/core/framework/op_expr_grad_function.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/common/container_util.h"

namespace oneflow {
namespace one {

struct AdaptiveAvgPoolNDGradGradCaptureState : public AutoGradCaptureState {
  bool input_requires_grad = false;
  bool grad_requires_grad = false;
  std::vector<int64_t> pool_output_size;
};

template<int ndims>
class AdaptiveAvgPoolNdNdGradGrad
    : public OpExprGradFunction<AdaptiveAvgPoolNDGradGradCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override { return Maybe<void>::Ok(); }

  Maybe<void> Capture(AdaptiveAvgPoolNDGradGradCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    // dy, x
    CHECK_EQ_OR_RETURN(inputs.size(), 2);   // NOLINT(maybe-need-error-msg)
    CHECK_EQ_OR_RETURN(outputs.size(), 1);  // NOLINT(maybe-need-error-msg)

    ctx->grad_requires_grad = inputs[0]->requires_grad();
    ctx->input_requires_grad = inputs[1]->requires_grad();
    if (ctx->grad_requires_grad) {
      const auto& grad_shape = *inputs[0]->shape();
      if (ndims == 1) {
        ctx->pool_output_size = {grad_shape[grad_shape.size() - 1]};
      } else if (ndims == 2) {
        ctx->pool_output_size = {grad_shape[grad_shape.size() - 2],
                                 grad_shape[grad_shape.size() - 1]};
      } else if (ndims == 3) {
        ctx->pool_output_size = {grad_shape[grad_shape.size() - 3],
                                 grad_shape[grad_shape.size() - 2],
                                 grad_shape[grad_shape.size() - 1]};
      } else {
        UNIMPLEMENTED_THEN_RETURN();
      }
    }
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const AdaptiveAvgPoolNDGradGradCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)
    in_grads->resize(2);

    if (ctx->grad_requires_grad) {
      if (ndims == 1) {
        (*in_grads)[0] = JUST(functional::AdaptiveAvgPool1D(out_grads[0], ctx->pool_output_size));
      } else if (ndims == 2) {
        (*in_grads)[0] = JUST(functional::AdaptiveAvgPool2D(out_grads[0], ctx->pool_output_size));
      } else if (ndims == 3) {
        (*in_grads)[0] = JUST(functional::AdaptiveAvgPool3D(out_grads[0], ctx->pool_output_size));
      } else {
        UNIMPLEMENTED_THEN_RETURN();
      }
    }
    if (ctx->input_requires_grad) { (*in_grads)[1] = JUST(functional::ZerosLike(out_grads[0])); }
    return Maybe<void>::Ok();
  }
};

struct AvgPoolGradGradCaptureState : public AutoGradCaptureState {
  bool input_requires_grad = false;
  bool grad_requires_grad = false;

  std::string data_format;
  std::vector<int32_t> padding;
  std::vector<int32_t> kernel_size;
  std::vector<int32_t> stride;
  bool ceil_mode = false;
  bool count_include_pad = false;
  int32_t divisor_override = 0;
};

class AvgPoolNdGradGrad : public OpExprGradFunction<AvgPoolGradGradCaptureState> {
 public:
  virtual ~AvgPoolNdGradGrad() = default;
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
    base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
    return Maybe<void>::Ok();
  }
  Maybe<void> Capture(AvgPoolGradGradCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    // dy, x
    CHECK_EQ_OR_RETURN(inputs.size(), 2);   // NOLINT(maybe-need-error-msg)
    CHECK_EQ_OR_RETURN(outputs.size(), 1);  // NOLINT(maybe-need-error-msg)

    ctx->grad_requires_grad = inputs[0]->requires_grad();
    ctx->input_requires_grad = inputs[1]->requires_grad();

    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    ctx->data_format = JUST(composed_attrs.GetAttr<std::string>("data_format"));
    ctx->padding = JUST(composed_attrs.GetAttr<std::vector<int32_t>>("padding"));
    ctx->kernel_size = JUST(composed_attrs.GetAttr<std::vector<int32_t>>("kernel_size"));
    ctx->stride = JUST(composed_attrs.GetAttr<std::vector<int32_t>>("stride"));
    ctx->ceil_mode = JUST(composed_attrs.GetAttr<bool>("ceil_mode"));
    ctx->count_include_pad = JUST(composed_attrs.GetAttr<bool>("count_include_pad"));
    ctx->divisor_override = JUST(composed_attrs.GetAttr<int32_t>("divisor_override"));

    return Maybe<void>::Ok();
  }
  Maybe<void> Apply(const AvgPoolGradGradCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)
    in_grads->resize(2);

    if (ctx->grad_requires_grad) {
      int32_t ndims = ctx->kernel_size.size();
      const auto pool_op =
          (ndims == 1 ? functional::AvgPool1D
                      : (ndims == 2 ? functional::AvgPool2D
                                    : (ndims == 3 ? functional::AvgPool3D : nullptr)));
      CHECK_NOTNULL_OR_RETURN(pool_op);  // NOLINT(maybe-need-error-msg)
      (*in_grads)[0] =
          JUST(pool_op(out_grads[0], ctx->kernel_size, ctx->stride, ctx->padding, ctx->ceil_mode,
                       ctx->count_include_pad, ctx->divisor_override, ctx->data_format));
    }
    if (ctx->input_requires_grad) { (*in_grads)[1] = JUST(functional::ZerosLike(out_grads[0])); }

    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("avg_pool_1d_grad", AvgPoolNdGradGrad);
REGISTER_OP_EXPR_GRAD_FUNCTION("avg_pool_2d_grad", AvgPoolNdGradGrad);
REGISTER_OP_EXPR_GRAD_FUNCTION("avg_pool_3d_grad", AvgPoolNdGradGrad);
REGISTER_OP_EXPR_GRAD_FUNCTION("adaptive_avg_pool1d_grad", AdaptiveAvgPoolNdNdGradGrad<1>);
REGISTER_OP_EXPR_GRAD_FUNCTION("adaptive_avg_pool2d_grad", AdaptiveAvgPoolNdNdGradGrad<2>);
REGISTER_OP_EXPR_GRAD_FUNCTION("adaptive_avg_pool3d_grad", AdaptiveAvgPoolNdNdGradGrad<3>);
}  // namespace one
}  // namespace oneflow
