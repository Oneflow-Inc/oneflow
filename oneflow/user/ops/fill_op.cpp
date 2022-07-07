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

/* static */ Maybe<void> FillOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& in_shape = ctx->InputShape("in", 0);
  Shape* out_shape = ctx->OutputShape("out", 0);
  *out_shape = in_shape;
  Stride* out_stride = ctx->OutputStride("out", 0);
  *out_stride = ctx->InputStride("in", 0);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> FillOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> FillOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& in_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
  FOR_RANGE(int64_t, i, 0, in_tensor.shape().NumAxes()) {
    ctx->NewBuilder().Split(user_op::OpArg("in", 0), i).Split(user_op::OpArg("out", 0), i).Build();
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> FillOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("out", 0) = ctx->InputDType("in", 0);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> FillTensorOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& in_shape = ctx->InputShape("in", 0);
  Shape* out_shape = ctx->OutputShape("out", 0);
  *out_shape = in_shape;
  Stride* out_stride = ctx->OutputStride("out", 0);
  *out_stride = ctx->InputStride("in", 0);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> FillTensorOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> FillTensorOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& in_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
  FOR_RANGE(int64_t, i, 0, in_tensor.shape().NumAxes()) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("in", 0), i)
        .Broadcast(user_op::OpArg("value", 0))
        .Split(user_op::OpArg("out", 0), i)
        .Build();
  }
  ctx->NewBuilder()
      .PartialSum(user_op::OpArg("in", 0))
      .PartialSum(user_op::OpArg("value", 0))
      .PartialSum(user_op::OpArg("out", 0))
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> FillTensorOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("out", 0) = ctx->InputDType("in", 0);
  return Maybe<void>::Ok();
}

REGISTER_USER_OP_GRAD("fill_").SetGenBackwardOpConfFn(
    [](const user_op::UserOpWrapper& op, const user_op::AddOpFn& AddOp) -> Maybe<void> {
      if (op.NeedGenGradTensor4OpInput("in", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name());
        user_op::UserOpConfWrapper grad_op =
            builder.Op("fill_")
                .Input("in", op.GetGradTensorWithOpOutput("out", 0))
                .Output("out")
                .Attr<double>("floating_value", 0.)
                .Attr<bool>("is_floating_value", true)
                .Build();
        op.BindGradTensorWithOpInput(grad_op.output("out", 0), "in", 0);
        AddOp(grad_op);
      }
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("fill_tensor_")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                               const user_op::AddOpFn& AddOp) -> Maybe<void> {
      if (op.NeedGenGradTensor4OpInput("value", 0)) {
        const int64_t num_axes = op.TensorDesc4ArgNameAndIndex("in", 0).shape().NumAxes();
        std::vector<int32_t> axes_vec(num_axes);
        std::iota(axes_vec.begin(), axes_vec.end(), 0);
        user_op::UserOpConfWrapperBuilder builder(op.op_name());
        auto grad_op = builder.Op("reduce_sum")
                           .Input("input_tensor", op.GetGradTensorWithOpOutput("out", 0))
                           .Output("output_tensor")
                           .Attr("axis", axes_vec)
                           .Attr("keepdims", false)
                           .Build();
        op.BindGradTensorWithOpInput(grad_op.output("out", 0), "value", 0);
        AddOp(grad_op);
      }
      if (op.NeedGenGradTensor4OpInput("in", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name());
        user_op::UserOpConfWrapper grad_op =
            builder.Op("fill_")
                .Input("in", op.GetGradTensorWithOpOutput("out", 0))
                .Output("out")
                .Attr<double>("floating_value", 0.)
                .Attr<bool>("is_floating_value", true)
                .Build();
        op.BindGradTensorWithOpInput(grad_op.output("out", 0), "in", 0);
        AddOp(grad_op);
      }
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
