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

/*static*/ Maybe<void> SplitLikeOp::GetSbp(user_op::SbpContext* ctx) {
  const auto axis = ctx->Attr<int64_t>("axis");
  const int64_t in_num_axes =
      ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0).shape().NumAxes();
  const int64_t like_num_axes =
      ctx->LogicalTensorDesc4InputArgNameAndIndex("like", 0).shape().NumAxes();
  FOR_RANGE(int64_t, i, 0, like_num_axes) {
    if (i == axis) { continue; }
    ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
  }
  std::vector<user_op::OpArg> like_arg_vec;
  const size_t like_arg_size = ctx->outputs().size();
  like_arg_vec.reserve(like_arg_size);
  FOR_RANGE(int32_t, i, 0, like_arg_size) { like_arg_vec.emplace_back("like", i); }
  FOR_RANGE(int64_t, i, like_num_axes, in_num_axes) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("in", 0), i)
        .Broadcast(like_arg_vec)
        .Split(ctx->outputs(), i)
        .Build();
    ctx->NewBuilder()
        .Split(user_op::OpArg("in", 0), i)
        .PartialSum(like_arg_vec)
        .Split(ctx->outputs(), i)
        .Build();
  }
  ctx->NewBuilder()
      .PartialSum(user_op::OpArg("in", 0))
      .PartialSum(like_arg_vec)
      .PartialSum(ctx->outputs())
      .Build();
  ctx->NewBuilder()
      .PartialSum(user_op::OpArg("in", 0))
      .Broadcast(like_arg_vec)
      .PartialSum(ctx->outputs())
      .Build();
  ctx->NewBuilder()
      .Broadcast(user_op::OpArg("in", 0))
      .PartialSum(like_arg_vec)
      .Broadcast(ctx->outputs())
      .Build();
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> SplitLikeOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const auto axis = ctx->Attr<int64_t>("axis");
  const user_op::TensorDesc& in_desc = ctx->InputTensorDesc("in", 0);
  int64_t dynamic_dim_size = 0;
  int64_t static_dim_size = 0;
  const int64_t in_num_axes = ctx->InputTensorDesc("in", 0).shape().NumAxes();
  const int64_t like_num_axes = ctx->InputTensorDesc("like", 0).shape().NumAxes();
  CHECK_LE_OR_RETURN(like_num_axes, in_num_axes)
      << Error::RuntimeError() << "The dimension of like (" << like_num_axes
      << ") should be less than or equal to input (" << in_num_axes << ")";
  CHECK_LT_OR_RETURN(axis, like_num_axes)
      << Error::RuntimeError() << "The axis (" << axis
      << ") should be less than the dimension of like (" << like_num_axes << ")";
  FOR_RANGE(int32_t, i, 0, ctx->outputs().size()) {
    const user_op::TensorDesc& like_i_desc = ctx->InputTensorDesc("like", i);
    user_op::TensorDesc* out_i_desc = ctx->MutOutputTensorDesc("out", i);
    CHECK_EQ_OR_RETURN(like_i_desc.shape().NumAxes(), like_num_axes)
        << Error::RuntimeError() << "The dimension of like_i (" << like_i_desc.shape().NumAxes()
        << ") must match the dimension of the first like (" << like_num_axes << ")";
    FOR_RANGE(int64_t, j, 0, like_num_axes) {
      if (j == axis) {
        if (like_i_desc.is_dynamic()) {
          dynamic_dim_size += like_i_desc.shape().At(j);
        } else {
          static_dim_size += like_i_desc.shape().At(j);
        }
      } else {
        CHECK_EQ_OR_RETURN(in_desc.shape().At(j), like_i_desc.shape().At(j))
            << Error::RuntimeError() << "The size of input (" << in_desc.shape().At(j)
            << ") must match the size of like_i (" << like_i_desc.shape().At(j) << ") at dimension "
            << j;
      }
    }
    DimVector out_i_dim_vec = like_i_desc.shape().dim_vec();
    FOR_RANGE(int64_t, j, like_num_axes, in_num_axes) {
      out_i_dim_vec.emplace_back(in_desc.shape().At(j));
    }
    out_i_desc->set_shape(Shape(out_i_dim_vec));
    out_i_desc->set_is_dynamic(like_i_desc.is_dynamic());
  }
  if (dynamic_dim_size == 0) {
    CHECK_EQ_OR_RETURN(static_dim_size, in_desc.shape().At(axis))
        << Error::RuntimeError() << "In non-dynamic shape situation, the total size of like ("
        << static_dim_size << ") should be equal to the size of input (" << in_desc.shape().At(axis)
        << ") at dimension " << axis;
  } else {
    CHECK_LE_OR_RETURN(static_dim_size, in_desc.shape().At(axis))
        << Error::RuntimeError() << "In dynamic shape situation, the total size of like ("
        << static_dim_size << ") should be less than or equal to the size of input ("
        << in_desc.shape().At(axis) << ") at dimension " << axis;
  }
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> SplitLikeOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> SplitLikeOp::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& in_desc = ctx->InputTensorDesc("in", 0);
  FOR_RANGE(int32_t, i, 0, ctx->outputs().size()) {
    user_op::TensorDesc* out_i_desc = ctx->MutOutputTensorDesc("out", i);
    out_i_desc->set_data_type(in_desc.data_type());
  }
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> SplitLikeOp::ModifyInputArg(const GetInputArgModifier& GetInputArgModifierFn,
                                                   const user_op::UserOpConfWrapper& user_op_conf) {
  FOR_RANGE(int32_t, i, 0, user_op_conf.input_size("like")) {
    user_op::InputArgModifier* like_modifier = GetInputArgModifierFn("like", i);
    CHECK_NOTNULL_OR_RETURN(like_modifier);  // NOLINT(maybe-need-error-msg)
    like_modifier->set_requires_grad(false);
  }
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> SplitLikeOp::CheckAttr(const user_op::UserOpDefWrapper&,
                                              const user_op::UserOpConfWrapper& op_conf) {
  CHECK_OR_RETURN(op_conf.input_size("like") >= 1)
      << Error::RuntimeError() << "The number of like should be greater than or equal to 1";
  CHECK_OR_RETURN(op_conf.output_size("out") >= 1)
      << Error::RuntimeError() << "The number of output should be greater than or equal to 1";
  return Maybe<void>::Ok();
}

}  // namespace oneflow
