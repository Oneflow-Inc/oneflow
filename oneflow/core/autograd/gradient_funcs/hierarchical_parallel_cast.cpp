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
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr_grad_function.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/nd_sbp.h"

namespace oneflow {

namespace one {

namespace {

Maybe<one::UserOpExpr> FindOrCreatHierarchicalParallelCastOpExpr(
    Symbol<cfg::ParallelDistribution> parallel_distribution) {
  thread_local HashMap<Symbol<cfg::ParallelDistribution>, std::shared_ptr<one::UserOpExpr>>
      parallel_distribution2hierarchical_parallel_cast_op_expr;
  auto iter = parallel_distribution2hierarchical_parallel_cast_op_expr.find(parallel_distribution);
  if (iter == parallel_distribution2hierarchical_parallel_cast_op_expr.end()) {
    std::shared_ptr<UserOpExpr> op_expr =
        JUST(OpBuilder("hierarchical_parallel_cast",
                       *CHECK_JUST(UniqueStr("hierarchical_parallel_cast")))
                 .Input("in")
                 .Output("out")
                 .Attr<std::vector<std::string>>("parallel_distribution",
                                                 *JUST(GetDualNdSbpStrList(parallel_distribution)))
                 .Attr<std::string>("grad_mode", "restore")
                 .Attr<std::vector<std::string>>("grad_parallel_distribution",
                                                 std::vector<std::string>())
                 .Build());
    iter = parallel_distribution2hierarchical_parallel_cast_op_expr
               .emplace(parallel_distribution, op_expr)
               .first;
  }
  return iter->second;
}

}  // namespace

struct HerarchicalParallelCastOpExprInterpState : public OpExprInterpState {
  Symbol<cfg::ParallelDistribution> parallel_distribution;
};

class HerarchicalParallelCast
    : public OpExprGradFunction<HerarchicalParallelCastOpExprInterpState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(HerarchicalParallelCastOpExprInterpState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    ctx->parallel_distribution = JUST(inputs.at(0)->parallel_distribution());
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const HerarchicalParallelCastOpExprInterpState* ctx,
                    const TensorTuple& out_grads, TensorTuple* in_grads) const override {
    const auto& grad_op =
        JUST(FindOrCreatHierarchicalParallelCastOpExpr(ctx->parallel_distribution));
    in_grads->resize(1);
    in_grads->at(0) = JUST(OpInterpUtil::Dispatch<Tensor>(*grad_op, {out_grads.at(0)}));
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("hierarchical_parallel_cast", HerarchicalParallelCast);

}  // namespace one
}  // namespace oneflow
