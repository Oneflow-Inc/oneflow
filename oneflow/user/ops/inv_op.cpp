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
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

/*static*/ Maybe<void> InvOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  *ctx->OutputShape("y", 0) = ctx->InputShape("x", 0);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> InvOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/*static*/ Maybe<void> InvOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& x = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
  FOR_RANGE(int64_t, i, 0, x.shape().NumAxes() - 2) {
    ctx->NewBuilder().Split(user_op::OpArg("x", 0), i).Split(user_op::OpArg("y", 0), i).Build();
  }
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> InvOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("y", 0) = ctx->InputDType("x", 0);
  return Maybe<void>::Ok();
}

Maybe<void> GenerateBackwardOpConf4Inv(const user_op::UserOpWrapper& op,
                                       const user_op::AddOpFn& AddOp) {
  if (op.NeedGenGradTensor4OpInput("x", 0)) {
    const auto& x = op.arg_tensor_desc("x", 0);
    const int64_t ndim = x.shape().NumAxes();
    std::string matmul_op_name("matmul");
    if (ndim > 2) { matmul_op_name = "batch_matmul"; }

    user_op::UserOpConfWrapperBuilder matmul_grad_builder(op.op_name() + "_grad_matmul_grad");
    user_op::UserOpConfWrapper matmul_grad_op =
        matmul_grad_builder.Op(matmul_op_name)
            .Input("a", op.GetGradTensorWithOpOutput("y", 0))
            .Input("b", op.output("y", 0))
            .Attr("transpose_a", false)
            .Attr("transpose_b", true)
            .Attr("alpha", 1.0)
            .Output("out")
            .Build();
    AddOp(matmul_grad_op);

    user_op::UserOpConfWrapperBuilder matmul_out_builder(op.op_name() + "_grad_matmul_out");
    user_op::UserOpConfWrapper matmul_out_op = matmul_out_builder.Op(matmul_op_name)
                                                   .Input("a", op.output("y", 0))
                                                   .Input("b", matmul_grad_op.output("out", 0))
                                                   .Attr("transpose_a", true)
                                                   .Attr("transpose_b", false)
                                                   .Attr("alpha", 1.0)
                                                   .Output("out")
                                                   .Build();
    AddOp(matmul_out_op);

    user_op::UserOpConfWrapperBuilder negative_builder(op.op_name() + "_grad_negative");
    user_op::UserOpConfWrapper negative_op = negative_builder.Op("negative")
                                                 .Input("x", matmul_out_op.output("out", 0))
                                                 .Output("y")
                                                 .Build();
    AddOp(negative_op);
    op.BindGradTensorWithOpInput(negative_op.output("y", 0), "x", 0);
  }
  return Maybe<void>::Ok();
}

REGISTER_USER_OP_GRAD("inv").SetGenBackwardOpConfFn(GenerateBackwardOpConf4Inv);

}  // namespace oneflow
