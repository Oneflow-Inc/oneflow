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

struct DimScatterCaptureState : public AutoGradCaptureState {
  int32_t dim;
  bool input_requires_grad;
  bool src_requires_grad;
};

enum SCATTER_TYPE { SCATTER_UPDATE, SCATTER_ADD };

template<SCATTER_TYPE T>
class DimScatter : public OpExprGradFunction<DimScatterCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(DimScatterCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const DimScatterCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;
  Maybe<void> ApplyCommon(const DimScatterCaptureState* ctx, const TensorTuple& out_grads,
                          TensorTuple* in_grads) const;

 private:
  AttrMap base_attrs_;
};

template<SCATTER_TYPE T>
Maybe<void> DimScatter<T>::Init(const OpExpr& op) {
  const UserOpExpr* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  return Maybe<void>::Ok();
}

template<SCATTER_TYPE T>
Maybe<void> DimScatter<T>::Capture(DimScatterCaptureState* ctx, const TensorTuple& inputs,
                                   const TensorTuple& outputs, const AttrMap& attrs) const {
  CHECK_EQ_OR_RETURN(inputs.size(), 3);   // NOLINT(maybe-need-error-msg)
  CHECK_EQ_OR_RETURN(outputs.size(), 1);  // NOLINT(maybe-need-error-msg)

  ctx->input_requires_grad = inputs.at(0)->requires_grad();
  ctx->src_requires_grad = inputs.at(2)->requires_grad();
  if ((!ctx->input_requires_grad) && (!ctx->src_requires_grad)) { return Maybe<void>::Ok(); }

  ctx->SaveTensorForBackward(inputs.at(1));  // index saved

  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->dim = JUST(composed_attrs.GetAttr<int32_t>("dim"));
  return Maybe<void>::Ok();
}

template<SCATTER_TYPE T>
Maybe<void> DimScatter<T>::ApplyCommon(const DimScatterCaptureState* ctx,
                                       const TensorTuple& out_grads, TensorTuple* in_grads) const {
  const std::shared_ptr<oneflow::one::Tensor>& index = ctx->SavedTensors().at(0);

  in_grads->resize(3);

  if (ctx->src_requires_grad) {
    in_grads->at(2) = JUST(functional::DimGather(out_grads.at(0), ctx->dim, index, false));
  }
  return Maybe<void>::Ok();
}

template<>
Maybe<void> DimScatter<SCATTER_TYPE::SCATTER_UPDATE>::Apply(const DimScatterCaptureState* ctx,
                                                            const TensorTuple& out_grads,
                                                            TensorTuple* in_grads) const {
  if ((!ctx->input_requires_grad) && (!ctx->src_requires_grad)) { return Maybe<void>::Ok(); }
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)
  JUST(ApplyCommon(ctx, out_grads, in_grads));

  if (ctx->input_requires_grad) {
    const std::shared_ptr<oneflow::one::Tensor>& index = ctx->SavedTensors().at(0);
    in_grads->at(0) =
        JUST(functional::DimScatterUpdateScalar(out_grads.at(0), ctx->dim, index, 0.0f));
  }
  return Maybe<void>::Ok();
}

template<>
Maybe<void> DimScatter<SCATTER_TYPE::SCATTER_ADD>::Apply(const DimScatterCaptureState* ctx,
                                                         const TensorTuple& out_grads,
                                                         TensorTuple* in_grads) const {
  if ((!ctx->input_requires_grad) && (!ctx->src_requires_grad)) { return Maybe<void>::Ok(); }
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)

  JUST(ApplyCommon(ctx, out_grads, in_grads));

  if (ctx->input_requires_grad) { in_grads->at(0) = out_grads.at(0); }

  return Maybe<void>::Ok();
}

class DimScatterUpdateScalar : public OpExprGradFunction<DimScatterCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(DimScatterCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const DimScatterCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
};

Maybe<void> DimScatterUpdateScalar::Init(const OpExpr& op) {
  const UserOpExpr* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());

  return Maybe<void>::Ok();
}

Maybe<void> DimScatterUpdateScalar::Capture(DimScatterCaptureState* ctx, const TensorTuple& inputs,
                                            const TensorTuple& outputs,
                                            const AttrMap& attrs) const {
  CHECK_EQ_OR_RETURN(inputs.size(), 2);   // NOLINT(maybe-need-error-msg)
  CHECK_EQ_OR_RETURN(outputs.size(), 1);  // NOLINT(maybe-need-error-msg)

  ctx->input_requires_grad = inputs.at(0)->requires_grad();
  if (!ctx->input_requires_grad) { return Maybe<void>::Ok(); }

  ctx->SaveTensorForBackward(inputs.at(1));  // index saved

  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->dim = JUST(composed_attrs.GetAttr<int32_t>("dim"));
  return Maybe<void>::Ok();
}

Maybe<void> DimScatterUpdateScalar::Apply(const DimScatterCaptureState* ctx,
                                          const TensorTuple& out_grads,
                                          TensorTuple* in_grads) const {
  if (!ctx->input_requires_grad) { return Maybe<void>::Ok(); }
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)
  const std::shared_ptr<oneflow::one::Tensor>& index = ctx->SavedTensors().at(0);

  in_grads->resize(2);

  in_grads->at(0) =
      JUST(functional::DimScatterUpdateScalar(out_grads.at(0), ctx->dim, index, 0.0f));

  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("dim_scatter_update", DimScatter<SCATTER_TYPE::SCATTER_UPDATE>);
REGISTER_OP_EXPR_GRAD_FUNCTION("dim_scatter_add", DimScatter<SCATTER_TYPE::SCATTER_ADD>);
REGISTER_OP_EXPR_GRAD_FUNCTION("dim_scatter_update_scalar", DimScatterUpdateScalar);

}  // namespace one
}  // namespace oneflow
