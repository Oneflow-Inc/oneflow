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

struct VarianceState : public AutoGradCaptureState {
  VarianceState() : requires_grad(false), unbiased(true), keepdim(false), axis({}){};
  bool requires_grad;
  bool unbiased;
  bool keepdim;
  std::vector<int32_t> axis;
};

class Variance : public OpExprGradFunction<VarianceState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(VarianceState* ctx, const TensorTuple& inputs, const TensorTuple& outputs,
                      const AttrMap& attrs) const override;
  Maybe<void> Apply(const VarianceState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
};

Maybe<void> Variance::Init(const OpExpr& op) {
  const UserOpExpr* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  return Maybe<void>::Ok();
}

Maybe<void> Variance::Capture(VarianceState* ctx, const TensorTuple& inputs,
                              const TensorTuple& outputs, const AttrMap& attrs) const {
  CHECK_EQ_OR_RETURN(inputs.size(), 1);   // NOLINT(maybe-need-error-msg)
  CHECK_EQ_OR_RETURN(outputs.size(), 1);  // NOLINT(maybe-need-error-msg)
  ctx->requires_grad = inputs.at(0)->requires_grad();
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }
  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->keepdim = JUST(composed_attrs.GetAttr<bool>("keepdim"));
  ctx->unbiased = JUST(composed_attrs.GetAttr<bool>("unbiased"));
  ctx->axis = JUST(composed_attrs.GetAttr<std::vector<int32_t>>("dim"));
  ctx->SaveTensorForBackward(inputs.at(0));
  return Maybe<void>::Ok();
}

Maybe<void> Variance::Apply(const VarianceState* ctx, const TensorTuple& out_grads,
                            TensorTuple* in_grads) const {
  // TODO(): replace it using kernel
  const std::shared_ptr<oneflow::one::Tensor>& x = ctx->SavedTensors().at(0);
  DataType data_type = x->dtype()->data_type();
  CHECK_NE_OR_RETURN(data_type, DataType::kBFloat16)
      << Error::RuntimeError() << "Variance op not support backward for bfloat16 yet!";
  size_t correction = ctx->unbiased ? 1 : 0;
  size_t elem_cnt = 1;
  CHECK_OR_RETURN(ctx->axis.size() > 0)
      << Error::RuntimeError() << "The size of the axis must greater than 0, but got "
      << ctx->axis.size();
  for (const auto& item : ctx->axis) { elem_cnt *= x->shape()->At(item); }

  std::shared_ptr<Tensor> out_grad = out_grads.at(0);
  if (ctx->keepdim == false) {
    // for broadcast mul
    const std::shared_ptr<const Shape>& out_grad_shape = out_grad->shape();
    DimVector unsqueeze_vector(out_grad_shape->dim_vec());
    for (int i = 0; i < ctx->axis.size(); i++) {
      unsqueeze_vector.insert(unsqueeze_vector.begin() + ctx->axis.at(i), 1);
    }
    Shape unsqueeze_shape(unsqueeze_vector);
    CHECK_EQ_OR_RETURN(unsqueeze_shape.elem_cnt(), out_grad_shape->elem_cnt())
        << Error::RuntimeError()
        << "tensor size mismatch, expected tensor to have the same number of elements, but got "
        << unsqueeze_shape.elem_cnt() << " and " << out_grad_shape->elem_cnt()
        << " elements respectively";
    out_grad = JUST(functional::Reshape(out_grad, unsqueeze_shape));
  }

  in_grads->resize(1);
  in_grads->at(0) = JUST(functional::Mul(
      out_grad,
      JUST(functional::ScalarMul(
          Scalar(2.0 / (elem_cnt - correction)),
          JUST(functional::Sub(x, JUST(functional::ReduceMean(x, ctx->axis, /*keepdim=*/true)),
                               /*alpha=*/1.0, /*inplace=*/false))))));

  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("var", Variance);

}  // namespace one
}  // namespace oneflow
