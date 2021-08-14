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
#include "oneflow/core/framework/id_util.h"
#include "oneflow/core/framework/op_expr_grad_function.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/job/sbp_parallel.h"

namespace oneflow {
namespace one {

struct ConsistentToConsistentOpExprInterpState : public OpExprInterpState {
  Symbol<ParallelDesc> parallel_desc;
  bool identity_grad;
  Symbol<cfg::ParallelDistribution> nd_sbp;
};

class ConsistentToConsistent : public OpExprGradFunction<ConsistentToConsistentOpExprInterpState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const ConsistentToConsistentOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);
    const std::string& op_name = fw_op_expr->op_name();
    grad_op_ = JUST(one::ConsistentToConsistentOpExpr::New(GradientOpName(op_name)));
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(ConsistentToConsistentOpExprInterpState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs,
                      const OpExprInterpContext& interp_ctx) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);
    ctx->parallel_desc = JUST(inputs[0]->parallel_desc());
    ctx->identity_grad = JUST(interp_ctx.attrs.GetAttr<bool>("identity_grad"));
    if (!ctx->identity_grad) {
      const auto& grad_sbp_str_list =
          JUST(interp_ctx.attrs.GetAttr<std::vector<std::string>>("grad_sbp"));
      if (grad_sbp_str_list.size() > 0) {
        CHECK_EQ_OR_RETURN(grad_sbp_str_list.size(), ctx->parallel_desc->hierarchy()->NumAxes());
        cfg::ParallelDistribution nd_sbp;
        for (const auto& sbp_str : grad_sbp_str_list) {
          CHECK_OR_RETURN(ParseSbpParallelFromString(sbp_str, nd_sbp.add_sbp_parallel()));
        }
        // manual
        ctx->nd_sbp = SymbolOf(nd_sbp);
      } else {
        // restore
        ctx->nd_sbp = JUST(inputs[0]->nd_sbp());
      }
    }
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const ConsistentToConsistentOpExprInterpState* ctx,
                    const TensorTuple& out_grads, TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    in_grads->resize(1);
    if (ctx->identity_grad) {
      (*in_grads)[0] = out_grads[0];
    } else {
      (*in_grads)[0] = JUST(OpInterpUtil::Dispatch<Tensor>(
          *grad_op_, out_grads, OpExprInterpContext(AttrMap{}, ctx->parallel_desc, ctx->nd_sbp)));
    }
    return Maybe<void>::Ok();
  }

 private:
  std::shared_ptr<OpExpr> grad_op_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("consistent_to_consistent", ConsistentToConsistent);

}  // namespace one
}  // namespace oneflow
