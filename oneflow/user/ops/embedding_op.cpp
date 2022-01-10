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

/* static */ auto EmbeddingRenormOp::InferLogicalTensorDesc(user_op::InferContext* ctx) -> Maybe<void> {
  const user_op::TensorDesc& in = ctx->InputTensorDesc("in", 0);
  CHECK_GT_OR_RETURN(in.shape().NumAxes(), 0);
  const int64_t axis = ctx->Attr<int64_t>("axis");
  const user_op::TensorDesc& indices = ctx->InputTensorDesc("indices", 0);
  CHECK_GT_OR_RETURN(indices.shape().NumAxes(), 0);
  user_op::TensorDesc* out = ctx->OutputTensorDesc("out", 0);

  DimVector dim_vec;
  dim_vec.insert(dim_vec.end(), in.shape().dim_vec().cbegin(),
                in.shape().dim_vec().cbegin() + axis);
  dim_vec.insert(dim_vec.end(), indices.shape().dim_vec().cbegin(),
                indices.shape().dim_vec().cend());
  dim_vec.insert(dim_vec.end(), in.shape().dim_vec().cbegin() + axis + 1,
                in.shape().dim_vec().end());
  *out->mut_shape() = Shape(dim_vec);
  out->set_is_dynamic(indices.is_dynamic() || in.is_dynamic());
  return Maybe<void>::Ok();
}
/*static*/ auto EmbeddingRenormOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) -> Maybe<void> {
  return EmbeddingRenormOp::InferLogicalTensorDesc(ctx);
}
/*static*/ auto EmbeddingRenormOp::GetSbp(user_op::SbpContext* ctx) -> Maybe<void> {
  const int64_t in_num_axes =
      ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0).shape().NumAxes();
  const int64_t indices_num_axes =
      ctx->LogicalTensorDesc4InputArgNameAndIndex("indices", 0).shape().NumAxes();
  const int64_t gather_axis = ctx->Attr<int64_t>("axis");
  CHECK_GE_OR_RETURN(gather_axis, 0);
  CHECK_LT_OR_RETURN(gather_axis, in_num_axes);
  FOR_RANGE(int64_t, i, 0, indices_num_axes) {
    ctx->NewBuilder()
    .Split(user_op::OpArg("indices", 0), i)
    .Broadcast(user_op::OpArg("in", 0))
    .Split(user_op::OpArg("out", 0), gather_axis + i)
    .Build();
  }
  FOR_RANGE(int64_t, i, 0, in_num_axes) {
    if (i == gather_axis) {
      ctx->NewBuilder()
          .Broadcast(user_op::OpArg("indices", 0))
          .Split(user_op::OpArg("in", 0), i)
          .PartialSum(user_op::OpArg("out", 0))
          .Build();
    } else {
      ctx->NewBuilder()
          .Broadcast(user_op::OpArg("indices", 0))
          .Split(user_op::OpArg("in", 0), i)
          .Split(user_op::OpArg("out", 0), i < gather_axis ? i : i + indices_num_axes - 1)
          .Build();
    }
  }
  return Maybe<void>::Ok();
}
/*static*/ auto EmbeddingRenormOp::InferDataType(user_op::InferContext* ctx) -> Maybe<void> {
  const user_op::TensorDesc& in = ctx->InputTensorDesc("in", 0);
  const user_op::TensorDesc& indices = ctx->InputTensorDesc("indices", 0);
  user_op::TensorDesc* out = ctx->OutputTensorDesc("out", 0);
  CHECK_OR_RETURN(IsIndexDataType(indices.data_type()));
  *out->mut_data_type() = in.data_type();
  return Maybe<void>::Ok();
}

/* static */ auto EmbeddingOp::InferLogicalTensorDesc(user_op::InferContext* ctx) -> Maybe<void> {
  const Shape& weight_shape = ctx->InputShape("weight", 0);
  const Shape& indices_shape = ctx->InputShape("indices", 0);
  CHECK_EQ_OR_RETURN(weight_shape.NumAxes(), 2);

  DimVector out_vec;
  out_vec.insert(out_vec.end(), indices_shape.dim_vec().cbegin(), indices_shape.dim_vec().cend());
  out_vec.push_back(weight_shape.At(1));
      
  user_op::TensorDesc* out_desc = ctx->OutputTensorDesc("out", 0);
  *out_desc->mut_shape() = Shape(out_vec);
  return Maybe<void>::Ok();
}
/*static*/ auto EmbeddingOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) -> Maybe<void> {
  return EmbeddingOp::InferLogicalTensorDesc(ctx);
}
/*static*/ auto EmbeddingOp::GetSbp(user_op::SbpContext* ctx) -> Maybe<void> {
  const user_op::TensorDesc& weight_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("weight", 0);
  const user_op::TensorDesc& indices_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("indices", 0);
  const user_op::TensorDesc& out_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("out", 0);
  const int32_t padding_idx = ctx->Attr<int32_t>("padding_idx");
  const bool scale_grad_by_freq = ctx->Attr<bool>("scale_grad_by_freq");
      
  int32_t out_num_axes = out_tensor.shape().NumAxes();

  if(padding_idx < 0 && !scale_grad_by_freq){
    for(int32_t i = 0; i < out_num_axes-1; i++){
        ctx->NewBuilder()
        .Split(user_op::OpArg("indices", 0), i)
        .Broadcast(user_op::OpArg("weight", 0))
        .Split(user_op::OpArg("out", 0), i)
        .Build();
    }

  }
  return Maybe<void>::Ok();
}
/*static*/ auto EmbeddingOp::InferDataType(user_op::InferContext* ctx) -> Maybe<void> {
  *ctx->OutputDType("out", 0) = ctx->InputDType("weight", 0);
  return Maybe<void>::Ok();
}

/* static */ auto EmbeddingGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) -> Maybe<void> {
  const Shape& weight_shape = ctx->InputShape("weight", 0);
  user_op::TensorDesc* dx_desc = ctx->OutputTensorDesc("dx", 0);
  *dx_desc->mut_shape() = weight_shape;

  return Maybe<void>::Ok();
}
/*static*/ auto EmbeddingGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) -> Maybe<void> {
  return EmbeddingGradOp::InferLogicalTensorDesc(ctx);
}
/*static*/ auto EmbeddingGradOp::GetSbp(user_op::SbpContext* ctx) -> Maybe<void> {
  const user_op::TensorDesc& dy_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("dy", 0);
  const user_op::TensorDesc& weight_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("weight", 0);
  const user_op::TensorDesc& indices_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("indices", 0);
  const user_op::TensorDesc& dx_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("dx", 0);
  const int32_t padding_idx = ctx->Attr<int32_t>("padding_idx");
  const bool scale_grad_by_freq = ctx->Attr<bool>("scale_grad_by_freq");

  int32_t dy_num_axes = dy_tensor.shape().NumAxes();

  if(padding_idx < 0 && !scale_grad_by_freq){
    for(int32_t i = 0; i < dy_num_axes - 1; i++){
        ctx->NewBuilder()
        .Broadcast(user_op::OpArg("dy", 0))
        .Broadcast(user_op::OpArg("weight", 0))
        .Split(user_op::OpArg("indices", 0), i)
        .PartialSum(user_op::OpArg("dx", 0))
        .Build();
    }
  }
  return Maybe<void>::Ok();
}
/*static*/ auto EmbeddingGradOp::InferDataType(user_op::InferContext* ctx) -> Maybe<void> {
  *ctx->OutputDType("dx", 0) = ctx->InputDType("weight", 0);
  return Maybe<void>::Ok();
}

REGISTER_USER_OP_GRAD("embedding").SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                                                          user_op::AddOpFn AddOp) -> Maybe<void> {
  bool need_grad_weight = op.NeedGenGradTensor4OpInput("weight", 0);
  if (need_grad_weight) {
    user_op::UserOpConfWrapperBuilder in_grad_builder(op.op_name() + "_grad");
    user_op::UserOpConfWrapper in_grad_op =
        in_grad_builder.Op("embedding_grad")
            .Input("dy", op.GetGradTensorWithOpOutput("weight", 0))
            .Input("weight", op.input("weight", 0))
            .Input("indices", op.input("indices", 0))
            .Output("dx")
            .Attr("padding_idx", op.attr<int32_t>("padding_idx"))
            .Attr("scale_grad_by_freq", op.attr<bool>("scale_grad_by_freq"))
            .Build();
    op.BindGradTensorWithOpInput(in_grad_op.output("dx", 0), "weight", 0);
    AddOp(in_grad_op);
  }
  return Maybe<void>::Ok();
});

}
