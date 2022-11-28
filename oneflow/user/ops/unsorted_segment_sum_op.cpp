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

/*static*/ Maybe<void> UnsortedSegmentSumOp::GetSbp(user_op::SbpContext* ctx) {
  const int64_t data_num_axes =
      ctx->LogicalTensorDesc4InputArgNameAndIndex("data", 0).shape().NumAxes();
  const int64_t segment_ids_num_axes =
      ctx->LogicalTensorDesc4InputArgNameAndIndex("segment_ids", 0).shape().NumAxes();
  const int64_t axis = ctx->Attr<int64_t>("axis");
  FOR_RANGE(int64_t, i, 0, segment_ids_num_axes) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("segment_ids", 0), i)
        .Split(user_op::OpArg("data", 0), i + axis)
        .PartialSum(user_op::OpArg("out", 0))
        .Build();
  }
  FOR_RANGE(int64_t, i, 0, data_num_axes) {
    if (i >= axis && i < axis + segment_ids_num_axes) { continue; }
    const int64_t out_split_axis = (i < axis) ? i : i - segment_ids_num_axes + 1;
    if (out_split_axis == axis) { continue; }
    ctx->NewBuilder()
        .Broadcast(user_op::OpArg("segment_ids", 0))
        .Split(user_op::OpArg("data", 0), i)
        .Split(user_op::OpArg("out", 0), out_split_axis)
        .Build();
  }
  ctx->NewBuilder()
      .Broadcast(user_op::OpArg("segment_ids", 0))
      .PartialSum(user_op::OpArg("data", 0))
      .PartialSum(user_op::OpArg("out", 0))
      .Build();
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> UnsortedSegmentSumOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& data_shape = ctx->InputShape("data", 0);
  const int64_t axis = ctx->Attr<int64_t>("axis");
  const int64_t num_segments = ctx->Attr<int64_t>("num_segments");
  const Shape& segment_ids_shape = ctx->InputShape("segment_ids", 0);

  DimVector dim_vec;
  dim_vec.insert(dim_vec.end(), data_shape.dim_vec().cbegin(),
                 data_shape.dim_vec().cbegin() + axis);
  dim_vec.emplace_back(num_segments);
  dim_vec.insert(dim_vec.end(), data_shape.dim_vec().cbegin() + axis + segment_ids_shape.NumAxes(),
                 data_shape.dim_vec().end());
  ctx->SetOutputShape("out", 0, Shape(dim_vec));
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> UnsortedSegmentSumOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> UnsortedSegmentSumOp::InferDataType(user_op::InferContext* ctx) {
  CHECK_OR_RETURN(IsIndexDataType(ctx->InputDType("segment_ids", 0)));
  ctx->SetOutputDType("out", 0, ctx->InputDType("data", 0));
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> UnsortedSegmentSumOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper&) {
  user_op::InputArgModifier* segment_ids_modifier = GetInputArgModifierFn("segment_ids", 0);
  CHECK_NOTNULL_OR_RETURN(segment_ids_modifier);
  segment_ids_modifier->set_requires_grad(false);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> UnsortedSegmentSumLikeOp::GetSbp(user_op::SbpContext* ctx) {
  const int64_t data_num_axes =
      ctx->LogicalTensorDesc4InputArgNameAndIndex("data", 0).shape().NumAxes();
  const int64_t segment_ids_num_axes =
      ctx->LogicalTensorDesc4InputArgNameAndIndex("segment_ids", 0).shape().NumAxes();
  const int64_t axis = ctx->Attr<int64_t>("axis");
  FOR_RANGE(int64_t, i, 0, segment_ids_num_axes) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("segment_ids", 0), i)
        .Split(user_op::OpArg("data", 0), i + axis)
        .Broadcast(user_op::OpArg("like", 0))
        .PartialSum(user_op::OpArg("out", 0))
        .Build();
    ctx->NewBuilder()
        .Split(user_op::OpArg("segment_ids", 0), i)
        .Split(user_op::OpArg("data", 0), i + axis)
        .PartialSum(user_op::OpArg("like", 0))
        .PartialSum(user_op::OpArg("out", 0))
        .Build();
  }
  FOR_RANGE(int64_t, i, 0, data_num_axes) {
    if (i >= axis && i < axis + segment_ids_num_axes) { continue; }
    const int64_t out_split_axis = (i < axis) ? i : i - segment_ids_num_axes + 1;
    if (out_split_axis == axis) { continue; }
    ctx->NewBuilder()
        .Broadcast(user_op::OpArg("segment_ids", 0))
        .Split(user_op::OpArg("data", 0), i)
        .Split(user_op::OpArg("like", 0), out_split_axis)
        .Split(user_op::OpArg("out", 0), out_split_axis)
        .Build();
  }
  ctx->NewBuilder()
      .Broadcast(user_op::OpArg("segment_ids", 0))
      .PartialSum(user_op::OpArg("data", 0))
      .Broadcast(user_op::OpArg("like", 0))
      .PartialSum(user_op::OpArg("out", 0))
      .Build();
  ctx->NewBuilder()
      .Broadcast(user_op::OpArg("segment_ids", 0))
      .PartialSum(user_op::OpArg("data", 0))
      .PartialSum(user_op::OpArg("like", 0))
      .PartialSum(user_op::OpArg("out", 0))
      .Build();
  ctx->NewBuilder()
      .Broadcast(user_op::OpArg("segment_ids", 0))
      .Broadcast(user_op::OpArg("data", 0))
      .Split(user_op::OpArg("like", 0), axis)
      .Split(user_op::OpArg("out", 0), axis)
      .Build();
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> UnsortedSegmentSumLikeOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  const Shape& data_shape = ctx->InputShape("data", 0);
  const Shape& like_shape = ctx->InputShape("like", 0);
  const Shape& segment_ids_shape = ctx->InputShape("segment_ids", 0);
  const int64_t axis = ctx->Attr<int64_t>("axis");
  CHECK_GE_OR_RETURN(axis, 0);
  CHECK_LE_OR_RETURN(axis, like_shape.NumAxes());
  FOR_RANGE(int64_t, i, 0, axis) { CHECK_EQ_OR_RETURN(like_shape.At(i), data_shape.At(i)); }
  CHECK_EQ_OR_RETURN(data_shape.NumAxes() - segment_ids_shape.NumAxes() + 1, like_shape.NumAxes());
  FOR_RANGE(int64_t, i, axis + 1, like_shape.NumAxes()) {
    CHECK_EQ_OR_RETURN(like_shape.At(i), data_shape.At(i + segment_ids_shape.NumAxes() - 1));
  }
  ctx->SetOutputShape("out", 0, ctx->InputShape("like", 0));
  ctx->SetIsDynamic4ArgNameAndIndex("out", 0, ctx->InputIsDynamic("like", 0));
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> UnsortedSegmentSumLikeOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> UnsortedSegmentSumLikeOp::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& data = ctx->InputTensorDesc("data", 0);
  const user_op::TensorDesc& like = ctx->InputTensorDesc("like", 0);
  CHECK_EQ_OR_RETURN(data.data_type(), like.data_type())
      << "InferDataType Failed. Expected " << DataType_Name(like.data_type()) << ", but got "
      << DataType_Name(data.data_type());
  CHECK_OR_RETURN(IsIndexDataType(ctx->InputDType("segment_ids", 0)));
  ctx->SetOutputDType("out", 0, ctx->InputDType("data", 0));
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> UnsortedSegmentSumLikeOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper&) {
  user_op::InputArgModifier* segment_ids_modifier = GetInputArgModifierFn("segment_ids", 0);
  CHECK_NOTNULL_OR_RETURN(segment_ids_modifier);
  segment_ids_modifier->set_requires_grad(false);
  user_op::InputArgModifier* like_modifier = GetInputArgModifierFn("like", 0);
  CHECK_NOTNULL_OR_RETURN(like_modifier);
  like_modifier->set_requires_grad(false);
  return Maybe<void>::Ok();
}

}  // namespace oneflow
