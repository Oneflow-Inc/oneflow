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
#include <cstdint>
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/op_generated.h"
#include "oneflow/core/framework/tensor_desc.h"
#include "oneflow/core/framework/user_op_conf.h"

namespace oneflow {

/*static*/ Maybe<void> IndexSelectOp::GetSbp(user_op::SbpContext* ctx) {
  const int32_t dim = ctx->Attr<int32_t>("dim");
  ctx->NewBuilder().Split(user_op::OpArg("x", 0), dim).Split(user_op::OpArg("index", 0), 0).Build();
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> IndexSelectOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& x_desc = ctx->InputTensorDesc("x", 0);
  const user_op::TensorDesc& index_desc = ctx->InputTensorDesc("index", 0);
  user_op::TensorDesc* y_desc = ctx->OutputTensorDesc("y", 0);
  const int32_t dim = ctx->Attr<int32_t>("dim");
  const int64_t num_axes = x_desc.shape().NumAxes();
  std::vector<int64_t> out_shape(num_axes);
  for (int i = 0; i < num_axes; i++) {
    if (i == dim) {
      out_shape[i] = index_desc.shape().At(0);
    } else {
      out_shape[i] = x_desc.shape().At(i);
    }
  }
  *y_desc->mut_shape() = Shape(DimVector(out_shape.begin(), out_shape.end()));

  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> IndexSelectOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> IndexSelectOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("y", 0) = ctx->InputDType("x", 0);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> IndexSelectGradOp::GetSbp(user_op::SbpContext* ctx) {
  const int32_t dim = ctx->Attr<int32_t>("dim");
  ctx->NewBuilder()
      .Split(user_op::OpArg("dy", 0), dim)
      .Split(user_op::OpArg("x", 0), dim)
      .Split(user_op::OpArg("index", 0), 0)
      .Split(user_op::OpArg("dx", 0), dim)
      .Build();
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> IndexSelectGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  Shape* dx_shape = ctx->OutputShape("dx", 0);
  *dx_shape = ctx->InputShape("x", 0);
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> IndexSelectGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> IndexSelectGradOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("dx", 0) = ctx->InputDType("dy", 0);
  return Maybe<void>::Ok();
}

REGISTER_USER_OP_GRAD("index_select")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                               user_op::AddOpFn AddOp) -> Maybe<void> {
      if (op.NeedGenGradTensor4OpInput("x", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper grad_op = builder.Op("index_select_grad")
                                                 .Input("dy", op.GetGradTensorWithOpOutput("y", 0))
                                                 .Input("x", op.input("x", 0))
                                                 .Input("index", op.input("index", 0))
                                                 .Output("dx")
                                                 .Attr("dim", op.attr<int32_t>("dim"))
                                                 .Build();
        op.BindGradTensorWithOpInput(grad_op.output("dx", 0), "x", 0);
        AddOp(grad_op);
      }
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
