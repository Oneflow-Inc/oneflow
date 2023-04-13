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

/*static*/ auto GatherOp::InferLogicalTensorDesc(user_op::InferContext* ctx) -> Maybe<void> {
  const user_op::TensorDesc& in = ctx->InputTensorDesc("in", 0);
  CHECK_GT_OR_RETURN(in.shape().NumAxes(), 0);
  const int64_t axis = ctx->Attr<int64_t>("axis");
  const user_op::TensorDesc& indices = ctx->InputTensorDesc("indices", 0);
  // For 0-dim Tensor
  CHECK_GE_OR_RETURN(indices.shape().NumAxes(), 0);  // NOLINT
  user_op::TensorDesc* out = ctx->MutOutputTensorDesc("out", 0);

  DimVector dim_vec;
  dim_vec.insert(dim_vec.end(), in.shape().dim_vec().cbegin(),
                 in.shape().dim_vec().cbegin() + axis);
  dim_vec.insert(dim_vec.end(), indices.shape().dim_vec().cbegin(),
                 indices.shape().dim_vec().cend());
  dim_vec.insert(dim_vec.end(), in.shape().dim_vec().cbegin() + axis + 1,
                 in.shape().dim_vec().end());
  out->set_shape(Shape(dim_vec));
  out->set_is_dynamic(indices.is_dynamic() || in.is_dynamic());
  return Maybe<void>::Ok();
}
/*static*/ auto GatherOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) -> Maybe<void> {
  return GatherOp::InferLogicalTensorDesc(ctx);
}
/*static*/ auto GatherOp::ModifyInputArg(const user_op::GetInputArgModifier& GetInputArgModifierFn,
                                         const user_op::UserOpConfWrapper&) -> Maybe<void> {
  user_op::InputArgModifier* indices_modifier = GetInputArgModifierFn("indices", 0);
  CHECK_OR_RETURN(indices_modifier != nullptr);
  indices_modifier->set_requires_grad(false);
  return Maybe<void>::Ok();
}
/*static*/ auto GatherOp::GetSbp(user_op::SbpContext* ctx) -> Maybe<void> {
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
/*static*/ auto GatherOp::InferDataType(user_op::InferContext* ctx) -> Maybe<void> {
  const user_op::TensorDesc& in = ctx->InputTensorDesc("in", 0);
  const user_op::TensorDesc& indices = ctx->InputTensorDesc("indices", 0);
  user_op::TensorDesc* out = ctx->MutOutputTensorDesc("out", 0);
  CHECK_OR_RETURN(IsIndexDataType(indices.data_type()));
  out->set_data_type(in.data_type());
  return Maybe<void>::Ok();
}

}  // namespace oneflow
