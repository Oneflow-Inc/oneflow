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
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/boxing/eager_boxing_interpreter_mgr.h"
#include "oneflow/core/framework/tensor_rpc_util.h"
#include "oneflow/core/common/decorator.h"
#include "oneflow/core/common/functor_util.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct CastGlobalCaptureState : public AutoGradCaptureState {
  Symbol<ParallelDesc> parallel_desc;
  Symbol<NdSbp> nd_sbp;
  std::shared_ptr<const Shape> shape;
  Symbol<DType> dtype;
};

class CastToGlobal : public OpExprGradFunction<CastGlobalCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const CastToGlobalOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
    const std::string& op_name = fw_op_expr->op_name();
    grad_op_ = JUST(one::CastFromGlobalOpExpr::New(GradientOpName(op_name)));
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(CastGlobalCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs,
                      const OpExprInterpContext& interp_ctx) const override {
    ctx->parallel_desc = JUST(interp_ctx.parallel_desc);
    ctx->nd_sbp = JUST(GetDualNdSbp(JUST(interp_ctx.nd_sbp)));
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const CastGlobalCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)
    std::shared_ptr<Tensor> out_grad = out_grads.at(0);
    CHECK_OR_RETURN(out_grad->is_global())
        << Error::RuntimeError()
        << "Expected global tensor for cast_to_global but got local tensor";
    {
      Symbol<NdSbp> nd_sbp_constraint = ctx->nd_sbp;
      Symbol<ParallelDesc> parallel_desc_constraint = ctx->parallel_desc;
      out_grad = JUST(functional::ToGlobal(out_grad, parallel_desc_constraint,
                                           *JUST(GetSbpList(nd_sbp_constraint)), GetNoneSbpList(),
                                           /* check_meta */ false));
    }
    in_grads->at(0) = JUST(OpInterpUtil::Dispatch<Tensor>(*grad_op_, {out_grad}));
    return Maybe<void>::Ok();
  }

 private:
  std::shared_ptr<OpExpr> grad_op_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("cast_to_global", CastToGlobal);

class CastFromGlobal : public OpExprGradFunction<CastGlobalCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const CastFromGlobalOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);  // NOLINT(maybe-need-error-msg)
    const std::string& op_name = fw_op_expr->op_name();
    grad_op_ = JUST(one::CastToGlobalOpExpr::New(GradientOpName(op_name)));
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(CastGlobalCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    const auto& input = inputs.at(0);
    CHECK_OR_RETURN(input->is_global())
        << Error::RuntimeError()
        << "Expected global tensor for cast_from_global but got local tensor";
    ctx->parallel_desc = JUST(input->parallel_desc());
    ctx->nd_sbp = JUST(input->nd_sbp());
    ctx->shape = input->shape();
    ctx->dtype = input->dtype();
    return Maybe<void>::Ok();
  }

  struct CastFromGlobalGradAttr {
    Maybe<AttrMap> operator()(const Shape& shape, DataType dtype, bool sync_data) {
      MutableAttrMap attrs;
      JUST(attrs.SetAttr<Shape>("shape", shape));
      JUST(attrs.SetAttr<DataType>("dtype", dtype));
      JUST(attrs.SetAttr<bool>("sync_data", sync_data));
      return AttrMap(attrs);
    }
  };

  Maybe<void> Apply(const CastGlobalCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    const auto& dual_nd_sbp = JUST(GetDualNdSbp(ctx->nd_sbp));
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(CastFromGlobalGradAttr);
    const auto& attrs = JUST(GetAttrs(*ctx->shape, ctx->dtype->data_type(), true));
    in_grads->at(0) = JUST(OpInterpUtil::Dispatch<Tensor>(
        *grad_op_, {out_grads.at(0)},
        OpExprInterpContext(*attrs, ctx->parallel_desc, dual_nd_sbp)));
    return Maybe<void>::Ok();
  }

 private:
  std::shared_ptr<OpExpr> grad_op_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("cast_from_global", CastFromGlobal);

}  // namespace one
}  // namespace oneflow
