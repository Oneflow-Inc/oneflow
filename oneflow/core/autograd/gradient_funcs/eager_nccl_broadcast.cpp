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
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_expr_helper.h"

namespace oneflow {
namespace one {

struct EagerNcclBroadcastOpExprInterpState : public OpExprInterpState {};

class EagerNcclBroadcast : public OpExprGradFunction<EagerNcclBroadcastOpExprInterpState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);
    const std::string& op_name = fw_op_expr->op_name();
    const auto& attr_map = fw_op_expr->base_attrs();
    grad_op_ = JUST(op_expr_helper::EagerNcclReduce(
        JUST(attr_map.GetAttr<std::string>("parallel_conf")),
        JUST(attr_map.GetAttr<int64_t>("root")), GradientOpName(op_name)));
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(EagerNcclBroadcastOpExprInterpState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    // do nothing
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const EagerNcclBroadcastOpExprInterpState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    in_grads->resize(1);
    in_grads->at(0) = JUST(OpInterpUtil::Dispatch<Tensor>(*grad_op_, {out_grads.at(0)}));
    return Maybe<void>::Ok();
  }

 private:
  std::shared_ptr<OpExpr> grad_op_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("eager_nccl_broadcast", EagerNcclBroadcast);

}  // namespace one
}  // namespace oneflow
