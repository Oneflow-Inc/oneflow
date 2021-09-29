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
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

// FloorDiv derivatives function isn't exists. (author: zhengzekang)
struct ScalarFloorDivCaptureState : public AutoGradCaptureState {};

class ScalarFloorDiv : public OpExprGradFunction<ScalarFloorDivCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override { return Maybe<void>::Ok(); }

  Maybe<void> Capture(ScalarFloorDivCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const ScalarFloorDivCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    UNIMPLEMENTED_THEN_RETURN() << "RuntimeError: derivative for floor_divide is not implemented";
    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("scalar_floordiv", ScalarFloorDiv);

}  // namespace one
}  // namespace oneflow
