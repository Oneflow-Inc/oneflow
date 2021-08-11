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

namespace oneflow {
namespace one {

namespace {

Maybe<one::ConsistentToConsistentOpExpr> FindOrCreatConsistentToConsistentOpExpr(
    Symbol<cfg::ParallelDistribution> parallel_distribution) {
  thread_local HashMap<Symbol<cfg::ParallelDistribution>,
                       std::shared_ptr<one::ConsistentToConsistentOpExpr>>
      parallel_distribution2consistent_to_consistent_op_expr;
  auto iter = parallel_distribution2consistent_to_consistent_op_expr.find(parallel_distribution);
  if (iter == parallel_distribution2consistent_to_consistent_op_expr.end()) {
    const auto& op_expr = JUST(one::ConsistentToConsistentOpExpr::New(
        *JUST(UniqueStr("consistent_to_consistent")), parallel_distribution));
    iter = parallel_distribution2consistent_to_consistent_op_expr
               .emplace(parallel_distribution, op_expr)
               .first;
  }
  return iter->second;
}

}  // namespace

struct ConsistentToConsistentOpExprInterpState : public OpExprInterpState {
  Symbol<cfg::ParallelDistribution> parallel_distribution;
  Symbol<ParallelDesc> parallel_desc;
};

class ConsistentToConsistent : public OpExprGradFunction<ConsistentToConsistentOpExprInterpState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const ConsistentToConsistentOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(ConsistentToConsistentOpExprInterpState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs,
                      const OpExprInterpContext& interp_ctx) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);
    ctx->parallel_desc = JUST(inputs.at(0)->parallel_desc());
    ctx->parallel_distribution = JUST(GetDualNdSbp(JUST(inputs.at(0)->parallel_distribution())));
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const ConsistentToConsistentOpExprInterpState* ctx,
                    const TensorTuple& out_grads, TensorTuple* in_grads) const override {
    const auto& grad_op = JUST(FindOrCreatConsistentToConsistentOpExpr(ctx->parallel_distribution));
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    in_grads->resize(1);
    in_grads->at(0) = JUST(OpInterpUtil::Dispatch<Tensor>(
        *grad_op, out_grads, OpExprInterpContext(AttrMap{}, ctx->parallel_desc)));
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("consistent_to_consistent", ConsistentToConsistent);

}  // namespace one
}  // namespace oneflow
