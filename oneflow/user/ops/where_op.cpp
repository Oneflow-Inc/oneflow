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
#include "oneflow/core/framework/dtype.h"

namespace oneflow {

namespace {

Maybe<Shape> GetBroadcastShape(const Shape& a_shape, const Shape& b_shape) {
  if (a_shape == b_shape) { return a_shape; }
  Shape broadcast_shape = Shape::Ones(std::max(a_shape.NumAxes(), b_shape.NumAxes()));
  Shape a_extend_shape = CreateLeftExtendedShape(ShapeView(a_shape), broadcast_shape.NumAxes());
  Shape b_extend_shape = CreateLeftExtendedShape(ShapeView(b_shape), broadcast_shape.NumAxes());
  FOR_RANGE(int64_t, i, 0, broadcast_shape.NumAxes()) {
    CHECK_OR_RETURN(a_extend_shape.At(i) == 1 || b_extend_shape.At(i) == 1
                    || a_extend_shape.At(i) == b_extend_shape.At(i))
        << Error::RuntimeError() << "The size of tensor a (" << a_extend_shape.At(i)
        << ") must match the size of tensor b (" << b_extend_shape.At(i)
        << ") at non-singleton dimension " << i;
    broadcast_shape.Set(i, std::max(a_extend_shape.At(i), b_extend_shape.At(i)));
  }
  return broadcast_shape;
}

}  // namespace

/*static*/ Maybe<void> WhereOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& cond_shape = ctx->InputShape("condition", 0);
  const Shape& x_shape = ctx->InputShape("x", 0);
  const Shape& y_shape = ctx->InputShape("y", 0);

  Shape broadcast_shape = *JUST(GetBroadcastShape(x_shape, y_shape));
  broadcast_shape = *JUST(GetBroadcastShape(broadcast_shape, cond_shape));
  ctx->SetOutputShape("out", 0, broadcast_shape);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> WhereOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/*static*/ Maybe<void> WhereOp::InferDataType(user_op::InferContext* ctx) {
  DataType cond_dtype = ctx->InputDType("condition", 0);
  CHECK_OR_RETURN(IsBoolDataType(cond_dtype) || IsIntegralDataType(cond_dtype));
  DataType x_dtype = ctx->InputDType("x", 0);
  CHECK_EQ_OR_RETURN(x_dtype, ctx->InputDType("y", 0))
      << "InferDataType Failed. Expected " << DataType_Name(ctx->InputDType("y", 0)) << ", but got "
      << DataType_Name(x_dtype);
  ctx->SetOutputDType("out", 0, x_dtype);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> WhereOp::GetSbp(user_op::SbpContext* ctx) {
  const Shape& cond_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("condition", 0).shape();
  const Shape& x_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0).shape();
  const Shape& y_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("y", 0).shape();
  Shape broadcast_shape = *JUST(GetBroadcastShape(x_shape, y_shape));
  broadcast_shape = *JUST(GetBroadcastShape(broadcast_shape, cond_shape));
  const size_t ndim = broadcast_shape.size();

  std::vector<user_op::OpArg> broadcast_args;
  std::vector<user_op::OpArg> split_args;
  std::vector<int> split_dims;
  broadcast_args.reserve(3);
  split_args.reserve(3);
  split_dims.reserve(3);

  auto CheckArgCanSplit = [&](std::string&& arg_name, const int dim, const Shape& shape) {
    size_t ddiff = ndim - shape.size();
    int dim_size = (dim >= ddiff) ? shape[dim - ddiff] : 1;
    if (dim_size == 1) {
      broadcast_args.emplace_back(std::forward<decltype(arg_name)>(arg_name), 0);
    } else {
      split_args.emplace_back(std::forward<decltype(arg_name)>(arg_name), 0);
      split_dims.push_back(dim - ddiff);
    }
  };

  for (int i = 0; i < ndim; ++i) {
    if (broadcast_shape[i] == 1) { continue; }
    broadcast_args.clear();
    split_args.clear();
    split_dims.clear();
    CheckArgCanSplit("x", i, x_shape);
    CheckArgCanSplit("y", i, y_shape);
    CheckArgCanSplit("condition", i, cond_shape);

    auto builder = ctx->NewBuilder();
    builder.Broadcast(broadcast_args);
    for (int i = 0; i < split_args.size(); ++i) { builder.Split(split_args[i], split_dims[i]); }
    builder.Split(user_op::OpArg("out", 0), i);
    builder.Build();
  }

  ctx->NewBuilder()
      .Broadcast(user_op::OpArg("condition", 0))
      .PartialSum(user_op::OpArg("x", 0))
      .PartialSum(user_op::OpArg("y", 0))
      .PartialSum(user_op::OpArg("out", 0))
      .Build();

  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> WhereOp::ModifyInputArg(const GetInputArgModifier& fn,
                                               const user_op::UserOpConfWrapper& conf) {
  user_op::InputArgModifier* cond_arg_modifier = fn("condition", 0);
  cond_arg_modifier->set_requires_grad(false);
  return Maybe<void>::Ok();
}

}  // namespace oneflow
