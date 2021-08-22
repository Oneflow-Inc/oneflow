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

Maybe<one::UserOpExpr> FindOrCreatHierarchicalParallelCastOpExpr(Symbol<cfg::NdSbp> nd_sbp) {
  thread_local HashMap<Symbol<cfg::NdSbp>, std::shared_ptr<one::UserOpExpr>>
      nd_sbp2hierarchical_parallel_cast_op_expr;
  auto iter = nd_sbp2hierarchical_parallel_cast_op_expr.find(nd_sbp);
  if (iter == nd_sbp2hierarchical_parallel_cast_op_expr.end()) {
    std::shared_ptr<UserOpExpr> op_expr =
        JUST(OpBuilder("hierarchical_parallel_cast",
                       *CHECK_JUST(UniqueStr("hierarchical_parallel_cast")))
                 .Input("in")
                 .Output("out")
                 .Attr<std::vector<std::string>>("nd_sbp", *JUST(GetDualNdSbpStrList(nd_sbp)))
                 .Attr<std::string>("grad_mode", "restore")
                 .Attr<std::vector<std::string>>("grad_nd_sbp", std::vector<std::string>())
                 .Build());
    iter = nd_sbp2hierarchical_parallel_cast_op_expr.emplace(nd_sbp, op_expr).first;
  }
  return iter->second;
}

}  // namespace

struct HerarchicalParallelCastCaptureState : public AutoGradCaptureState {
  Symbol<cfg::NdSbp> nd_sbp;
};

class HerarchicalParallelCast : public OpExprGradFunction<HerarchicalParallelCastCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(HerarchicalParallelCastCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    ctx->nd_sbp = JUST(inputs.at(0)->nd_sbp());
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const HerarchicalParallelCastCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    const auto& grad_op = JUST(FindOrCreatHierarchicalParallelCastOpExpr(ctx->nd_sbp));
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    in_grads->resize(1);
    in_grads->at(0) = out_grads.at(0);
    JUST(OpInterpUtil::Dispatch(*grad_op, {out_grads.at(0)}, in_grads));
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("hierarchical_parallel_cast", HerarchicalParallelCast);

}  // namespace one
}  // namespace oneflow
