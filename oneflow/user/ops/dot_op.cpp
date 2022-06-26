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

/* static */ Maybe<void> DotOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& x = ctx->InputTensorDesc("x", 0);
  const user_op::TensorDesc& y = ctx->InputTensorDesc("y", 0);
  CHECK_OR_RETURN(x.shape() == y.shape())
      << Error::RuntimeError()
      << "inconsistent tensor size, expected tensor to have the same number of elements, but got "
      << x.shape().elem_cnt() << " and " << y.shape().elem_cnt() << " elements respectively";
  CHECK_OR_RETURN(x.shape().NumAxes() == 1)
      << Error::RuntimeError() << "1D tensors expected, but got " << x.shape().NumAxes()
      << "D tensors";
  *ctx->OutputShape("out", 0) = Shape({});
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> DotOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> DotOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Split(user_op::OpArg("x", 0), 0)
      .Split(user_op::OpArg("y", 0), 0)
      .PartialSum(user_op::OpArg("out", 0))
      .Build();

  return Maybe<void>::Ok();
}

/* static */ Maybe<void> DotOp::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& x = ctx->InputTensorDesc("x", 0);
  const user_op::TensorDesc& y = ctx->InputTensorDesc("y", 0);
  CHECK_OR_RETURN(x.data_type() == y.data_type())
      << Error::RuntimeError() << "expected both vectors to have same dtype, but found "
      << DataType_Name(x.data_type()) << " and " << DataType_Name(y.data_type());
  *ctx->OutputDType("out", 0) = ctx->InputDType("x", 0);
  return Maybe<void>::Ok();
}

REGISTER_USER_OP_GRAD("dot").SetGenBackwardOpConfFn(
    [](const user_op::UserOpWrapper& op, const user_op::AddOpFn& AddOp) -> Maybe<void> {
      if (op.NeedGenGradTensor4OpInput("x", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper grad_op =
            builder.Op("scalar_mul")
                .Input("x", op.input("y", 0))
                .Input("scalar", op.GetGradTensorWithOpOutput("out", 0))
                .Output("y")
                .Build();

        op.BindGradTensorWithOpInput(grad_op.output("y", 0), "x", 0);
      }

      if (op.NeedGenGradTensor4OpInput("y", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper grad_op =
            builder.Op("scalar_mul")
                .Input("x", op.input("x", 0))
                .Input("scalar", op.GetGradTensorWithOpOutput("out", 0))
                .Output("y")
                .Build();

        op.BindGradTensorWithOpInput(grad_op.output("y", 0), "y", 0);
      }
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
