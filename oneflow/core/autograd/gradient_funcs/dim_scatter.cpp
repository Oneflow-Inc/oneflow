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
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_expr_helper.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"

namespace oneflow {
namespace one {

struct DimScatterInterpState : public OpExprInterpState {
  int32_t dim;
  bool requires_input;
  bool requires_src;
};

class DimScatter : public OpExprGradFunction<DimScatterInterpState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(DimScatterInterpState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const DimScatterInterpState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  AttrMap base_attrs_;
  std::shared_ptr<OpExpr> dim_gather_op_;
  std::shared_ptr<OpExpr> dim_scatter_scalar_op_;
};

Maybe<void> DimScatter::Init(const OpExpr& op) {

  return Maybe<void>::Ok();
}

Maybe<void> DimScatter::Capture(DimScatterInterpState* ctx, const TensorTuple& inputs,
                               const TensorTuple& outputs, const AttrMap& attrs) const {

  return Maybe<void>::Ok();
}

Maybe<void> DimScatter::Apply(const DimScatterInterpState* ctx, const TensorTuple& out_grads,
                             TensorTuple* in_grads) const {
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("dim_scatter", DimScatter);

}  // namespace one
}  // namespace oneflow
