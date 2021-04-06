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
#include "oneflow/core/autograd/gradient_funcs/utility.h"
#include "oneflow/core/framework/op_expr_grad_function.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_dispatch.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_expr_helper.h"
#include "oneflow/core/framework/op_interpreter_util.h"

namespace oneflow {
namespace one {

class BiasAdd : public OpExprGradFunction {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);
    op_name_ = fw_op_expr->op_name();
    bias_add_axis_ = GetAttr<int32_t>(fw_op_expr->proto(), "axis");
    backward_input_op_ = JUST(op_expr_helper::IdentityOp(GradientOpName(op_name_ + "_input")));
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(OpExprInterpState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 2);
    input_requires_grad_ = inputs.at(0)->requires_grad();
    bias_requires_grad_ = inputs.at(1)->requires_grad();
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const OpExprInterpState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    const int64_t num_axes = out_grads.at(0)->shape()->NumAxes();
    if (bias_requires_grad_
        && (reduce_axes_vec_.size() != num_axes - 1 || !backward_bias_op_.get())) {
      reduce_axes_vec_.clear();
      for (int i = 0; i < num_axes; ++i) {
        if (i != bias_add_axis_) { reduce_axes_vec_.push_back(i); }
      }
      backward_bias_op_ = JUST(op_expr_helper::ReduceSumOp(reduce_axes_vec_, /*keepdims=*/false,
                                                           GradientOpName(op_name_ + "_bias")));
    }
    in_grads->resize(2);
    if (input_requires_grad_) {
      in_grads->at(0) = JUST(Dispatch<Tensor>(*backward_input_op_, {out_grads.at(0)}));
    }
    if (bias_requires_grad_) {
      in_grads->at(1) = JUST(Dispatch<Tensor>(*backward_bias_op_, {out_grads.at(0)}));
    }
    return Maybe<void>::Ok();
  }

 private:
  std::string op_name_;
  int32_t bias_add_axis_;
  mutable bool input_requires_grad_;
  mutable bool bias_requires_grad_;
  std::shared_ptr<OpExpr> backward_input_op_;
  mutable std::shared_ptr<OpExpr> backward_bias_op_;
  mutable std::vector<int32_t> reduce_axes_vec_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("bias_add", BiasAdd);

}  // namespace one
}  // namespace oneflow
