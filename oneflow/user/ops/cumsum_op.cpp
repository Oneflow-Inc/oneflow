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

REGISTER_USER_OP("cumsum")
    .Input("in")
    .Output("out")
    .Attr<int32_t>("axis")
    .Attr<bool>("exclusive")
    .Attr<bool>("reverse")
    .SetTensorDescInferFn(user_op::TensorDescInferFnUtil::Unchanged)
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& in_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
      const int32_t axis = ctx->Attr<int32_t>("axis");
      FOR_RANGE(int64_t, i, 0, in_tensor.shape().NumAxes()) {
        if (i != axis) {
          ctx->NewBuilder()
              .Split(user_op::OpArg("in", 0), i)
              .Split(user_op::OpArg("out", 0), i)
              .Build();
        }
      }

      ctx->NewBuilder()
          .PartialSum(user_op::OpArg("in", 0))
          .PartialSum(user_op::OpArg("out", 0))
          .Build();

      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("cumsum").SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                                                          user_op::AddOpFn AddOp) {
  if (op.NeedGenGradTensor4OpInput("in", 0)) {
    user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
    user_op::UserOpConfWrapper cumsum_grad_op =
        builder.Op("cumsum")
            .Input("in", op.GetGradTensorWithOpOutput("out", 0))
            .Output("out")
            .Attr("axis", op.attr<int32_t>("axis"))
            .Attr("exclusive", op.attr<bool>("exclusive"))
            .Attr("reverse", not op.attr<bool>("reverse"))
            .Build();
    op.BindGradTensorWithOpInput(cumsum_grad_op.output("out", 0), "in", 0);
    AddOp(cumsum_grad_op);
  }
});

}  // namespace oneflow
