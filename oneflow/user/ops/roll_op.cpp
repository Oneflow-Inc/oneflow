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

namespace {

REGISTER_USER_OP("roll")
    .Input("in")
    .Output("out")
    .Attr<std::vector<int32_t>>("shifts", std::vector<int32_t>{1})
    .Attr<std::vector<int32_t>>("dims", std::vector<int32_t>{0})
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputShape("out", 0) = ctx->InputShape("in", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
        const user_op::TensorDesc& in_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
        const std::vector<int32_t> dims_vector = ctx->Attr<std::vector<int32_t>>("dims");
        int64_t flag = 0;
        FOR_RANGE(int64_t, i, 0, in_tensor.shape().NumAxes() - 1) {
            if(std::find(dims_vector.begin(), dims_vector.end(), i) != dims_vector.end()) {
                flag = 1;
            }
            if(flag) continue;
            ctx->NewBuilder()
                .Split(user_op::OpArg("in", 0), i)
                .Split(user_op::OpArg("out", 0), i)
                .Build();
        }

        return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
        *ctx->OutputDType("out", 0) = ctx->InputDType("in", 0);
        return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("roll").SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                                                        user_op::AddOpFn AddOp) -> Maybe<void> {
    if(op.NeedGenGradTensor4OpInput("in", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper roll_grad_op = 
            builder.Op("roll_grad")
                .Input("in", op.GetGradTensorWithOpOutput("out", 0))
                .Attr<std::vector<int32_t>>("shifts", std::vector<int32_t>{1})
                .Attr<std::vector<int32_t>>("dims", std::vector<int32_t>{0})
                .Output("out")
                .Build();
        op.BindGradTensorWithOpInput(roll_grad_op.output("out", 0), "in", 0);
        AddOp(roll_grad_op);
    }
    return Maybe<void>::Ok();                                                      
    });

}   // namespace
}   // namespace oneflow