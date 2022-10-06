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
#include "oneflow/core/operator/reduce_sbp_util.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

namespace {

Maybe<void> GetSbpSignatures(user_op::SbpContext* ctx) {
  const auto& x_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0).shape();
  const auto& like_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("like", 0).shape();
  int32_t x_num_axes = x_shape.NumAxes();
  int32_t like_num_axes = like_shape.NumAxes();
  const auto& reduced_axes = ctx->Attr<std::vector<int32_t>>("broadcast_axes");
  HashSet<int32_t> conf_axes;
  ReduceSbpUtil::GetRegularAxes(like_num_axes, reduced_axes, &conf_axes);
  auto IsReducedAxis = ReduceSbpUtil::MakePredicatorIsReducedAxis(conf_axes, like_num_axes);
  int32_t num_reduced_axis = 0;
  FOR_RANGE(int64_t, i, 0, like_num_axes) {
    if (IsReducedAxis(i)) {
      ctx->NewBuilder()
          .Broadcast(user_op::OpArg("x", 0))
          .Split(user_op::OpArg("like", 0), i)
          .Split(user_op::OpArg("y", 0), i)
          .Build();
      if (x_num_axes < like_num_axes) { num_reduced_axis += 1; }
    } else {
      ctx->NewBuilder()
          .Split(user_op::OpArg("x", 0), i - num_reduced_axis)
          .Split(user_op::OpArg("like", 0), i)
          .Split(user_op::OpArg("y", 0), i)
          .Build();
    }
  }
  ctx->NewBuilder().PartialSum(ctx->inputs()).PartialSum(ctx->outputs()).Build();
  ctx->NewBuilder()
      .PartialSum(user_op::OpArg("x", 0))
      .Broadcast(user_op::OpArg("like", 0))
      .PartialSum(user_op::OpArg("y", 0))
      .Build();
  ctx->NewBuilder()
      .Broadcast(user_op::OpArg("x", 0))
      .PartialSum(user_op::OpArg("like", 0))
      .Broadcast(user_op::OpArg("y", 0))
      .Build();
  return Maybe<void>::Ok();
}

bool IsAxesLegal(const AxisVector& axis_vec, const Shape& like_shape, const Shape& in_shape) {
  Shape reduced_like_shape = CreateReducedShape(like_shape, axis_vec);
  if (like_shape.NumAxes() > in_shape.NumAxes()) {
    std::vector<int64_t> in_shape_vec;
    in_shape_vec.reserve(in_shape.NumAxes());
    std::vector<int64_t> like_shape_vec;
    like_shape_vec.reserve(reduced_like_shape.NumAxes());
    for (const int64_t& dim : in_shape.dim_vec()) {
      if (dim != 1) { in_shape_vec.emplace_back(dim); }
    }
    for (const int64_t& dim : reduced_like_shape.dim_vec()) {
      if (dim != 1) { like_shape_vec.emplace_back(dim); }
    }
    if (in_shape_vec.size() > like_shape_vec.size()) {
      return false;
    } else {
      return std::equal(in_shape_vec.begin(), in_shape_vec.end(), like_shape_vec.begin());
    }
  }
  return reduced_like_shape.dim_vec() == in_shape.dim_vec();
}

Maybe<void> InferTensorDesc(user_op::InferContext* ctx) {
  const auto& broadcast_axes = ctx->Attr<std::vector<int32_t>>("broadcast_axes");
  CHECK_OR_RETURN(!broadcast_axes.empty());
  const Shape& in_shape = ctx->InputShape("x", 0);
  const Shape& like_shape = ctx->InputShape("like", 0);
  const AxisVector axis_vec = {broadcast_axes.begin(), broadcast_axes.end()};
  CHECK_OR_RETURN(IsAxesLegal(axis_vec, like_shape, in_shape))
      << Error::RuntimeError() << "Invalid input parameter: like shape:" << like_shape.ToString()
      << ", in shape:" << in_shape.ToString() << ", axis_vec size:" << axis_vec.size();
  ctx->SetOutputShape("y", 0, like_shape);
  ctx->SetOutputStride("y", 0, Stride(like_shape));
  return Maybe<void>::Ok();
}

}  // namespace

/* static */ Maybe<void> BroadcastLikeOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return InferTensorDesc(ctx);
}

/*static*/ Maybe<void> BroadcastLikeOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> BroadcastLikeOp::GetSbp(user_op::SbpContext* ctx) {
  return GetSbpSignatures(ctx);
}

/* static */ Maybe<void> BroadcastLikeOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper& conf) {
  user_op::InputArgModifier* like_modifier = GetInputArgModifierFn("like", 0);
  CHECK_OR_RETURN(like_modifier != nullptr);
  like_modifier->set_requires_grad(false);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> BroadcastLikeOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("y", 0, ctx->InputDType("x", 0));
  return Maybe<void>::Ok();
}

}  // namespace oneflow
