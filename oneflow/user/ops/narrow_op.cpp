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

/* static */ Maybe<void> NarrowOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& in = ctx->InputTensorDesc("in", 0);
  CHECK_GT_OR_RETURN(in.shape().NumAxes(), 0);
  const int64_t& dim = ctx->Attr<int64_t>("dim");
  const int64_t& start = ctx->Attr<int64_t>("start");
  int64_t length = ctx->Attr<int64_t>("length");
  CHECK_GE_OR_RETURN(dim, 0);
  CHECK_GE_OR_RETURN(start, 0);
  CHECK_GE_OR_RETURN(length, 0);
  // length should be input size if split the full slice dimension
  if (start == 0 && length > in.shape().At(dim)) { length = in.shape().At(dim); }
  user_op::TensorDesc* out = ctx->MutOutputTensorDesc("out", 0);

  DimVector dim_vec;
  dim_vec.insert(dim_vec.end(), in.shape().dim_vec().cbegin(), in.shape().dim_vec().cbegin() + dim);
  dim_vec.insert(dim_vec.end(), length);
  dim_vec.insert(dim_vec.end(), in.shape().dim_vec().cbegin() + dim + 1,
                 in.shape().dim_vec().end());
  out->set_shape(Shape(dim_vec));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> NarrowOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> NarrowOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& in_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
  const int64_t& dim = ctx->Attr<int64_t>("dim");
  const int64_t& length = ctx->Attr<int64_t>("length");
  FOR_RANGE(int64_t, i, 0, in_tensor.shape().NumAxes()) {
    if (i != dim) {
      ctx->NewBuilder()
          .Split(user_op::OpArg("in", 0), i)
          .Split(user_op::OpArg("out", 0), i)
          .Build();
    } else {
      if (length == in_tensor.shape().At(i)) {
        ctx->NewBuilder()
            .Split(user_op::OpArg("in", 0), i)
            .Split(user_op::OpArg("out", 0), i)
            .Build();
      }
    }
  }
  ctx->NewBuilder()
      .PartialSum(user_op::OpArg("in", 0))
      .PartialSum(user_op::OpArg("out", 0))
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> NarrowOp::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& in = ctx->InputTensorDesc("in", 0);
  user_op::TensorDesc* out = ctx->MutOutputTensorDesc("out", 0);
  out->set_data_type(in.data_type());
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> NarrowGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& like_shape = ctx->InputShape("like", 0);
  const Shape& dy_shape = ctx->InputShape("dy", 0);
  const int64_t ndim = dy_shape.NumAxes();
  CHECK_EQ_OR_RETURN(like_shape.NumAxes(), ndim);

  ctx->SetOutputShape("dx", 0, like_shape);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> NarrowGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> NarrowGradOp::GetSbp(user_op::SbpContext* ctx) {
  const Shape& like_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("like", 0).shape();
  const int64_t ndim = like_shape.NumAxes();
  const int64_t& dim = ctx->Attr<int64_t>("dim");
  const int64_t& length = ctx->Attr<int64_t>("length");
  FOR_RANGE(int64_t, i, 0, ndim) {
    if (i != dim) {
      ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
    } else {
      if (length == like_shape.At(i)) {
        ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
      }
    }
  }
  ctx->NewBuilder().PartialSum(ctx->inputs()).PartialSum(ctx->outputs()).Build();
  ctx->NewBuilder()
      .PartialSum(user_op::OpArg("dy", 0))
      .Broadcast(user_op::OpArg("like", 0))
      .PartialSum(user_op::OpArg("dx", 0))
      .Build();
  ctx->NewBuilder()
      .Broadcast(user_op::OpArg("dy", 0))
      .PartialSum(user_op::OpArg("like", 0))
      .Broadcast(user_op::OpArg("dx", 0))
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> NarrowGradOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper& conf) {
  user_op::InputArgModifier* dy_modifier = GetInputArgModifierFn("dy", 0);
  CHECK_NOTNULL_OR_RETURN(dy_modifier);
  dy_modifier->set_requires_grad(false);
  user_op::InputArgModifier* like_modifier = GetInputArgModifierFn("like", 0);
  CHECK_NOTNULL_OR_RETURN(like_modifier);
  like_modifier->set_requires_grad(false);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> NarrowGradOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("dx", 0, ctx->InputDType("dy", 0));
  return Maybe<void>::Ok();
}

}  // namespace oneflow
