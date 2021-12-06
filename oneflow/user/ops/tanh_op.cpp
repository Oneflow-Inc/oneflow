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
#include "oneflow/core/framework/framework.h"

namespace oneflow {

REGISTER_USER_OP("tanh")
    .Input("x")
    .Output("y")
    .SetTensorDescInferFn(user_op::TensorDescInferFnUtil::Unchanged)
    .SetGetSbpFn(user_op::GetSbpFnUtil::SplitForEachAxis)
    .SetDataTypeInferFn(user_op::TensorDescInferFnUtil::UnchangedDataType);

REGISTER_USER_OP((std::string("") + "tanh" + "_grad"))
    .Input("x")
    .Input("dy")
    .Output("dx")
    .SetTensorDescInferFn(user_op::TensorDescInferFnUtil::Unchanged)
    .SetGetSbpFn(user_op::GetSbpFnUtil::SplitForEachAxis)
    .SetDataTypeInferFn(user_op::TensorDescInferFnUtil::UnchangedDataType);

REGISTER_USER_OP_GRAD("tanh").SetGenBackwardOpConfFn(
    [](const user_op::UserOpWrapper& op, const user_op::AddOpFn& AddOp) -> Maybe<void> {
      if (op.NeedGenGradTensor4OpInput("x", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper unary_grad_op =
            builder.Op((std::string("") + "tanh" + "_grad"))
                .Input("x", op.input("x", 0))
                .Input("dy", op.GetGradTensorWithOpOutput("y", 0))
                .Output("dx")
                .Build();
        op.BindGradTensorWithOpInput(unary_grad_op.output("dx", 0), "x", 0);
        AddOp(unary_grad_op);
      }
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
