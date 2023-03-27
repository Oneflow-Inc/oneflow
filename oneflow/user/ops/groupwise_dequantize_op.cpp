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

/*static*/ Maybe<void> GroupwiseDequantizeOp::GetSbp(user_op::SbpContext* ctx) {
  const Shape& in_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0).shape();
  const Shape& scale_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("scale", 0).shape();
  std::vector<user_op::OpArg> scale_zero_args;
  scale_zero_args.emplace_back(user_op::OpArg("scale", 0));
  if (ctx->user_op_conf().has_input("zero", 0)) {
    scale_zero_args.emplace_back(user_op::OpArg("zero", 0));
  }
  for (int32_t i = 0; i < in_shape.NumAxes(); ++i) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("in", 0), i)
        .Split(scale_zero_args, i)
        .Split(user_op::OpArg("out", 0), i)
        .Build();
  }
  const int64_t group_dim = ctx->Attr<int64_t>("group_dim");
  if (scale_shape.At(group_dim) == 1) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("in", 0), scale_shape.NumAxes())
        .Broadcast(scale_zero_args)
        .Split(user_op::OpArg("out", 0), scale_shape.NumAxes())
        .Build();
  }
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> GroupwiseDequantizeOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& in_shape = ctx->InputShape("in", 0);
  const Shape& scale_shape = ctx->InputShape("scale", 0);
  const int32_t num_bits = ctx->Attr<int32_t>("num_bits");
  const int64_t group_dim = ctx->Attr<int64_t>("group_dim");
  const int64_t group_size = ctx->Attr<int64_t>("group_size");
  CHECK_OR_RETURN(num_bits == 4 || num_bits == 8);
  CHECK_GE_OR_RETURN(in_shape.NumAxes(), 1);
  CHECK_OR_RETURN(group_dim >= 0 && group_dim < in_shape.NumAxes());
  Shape out_shape = in_shape;
  out_shape.Set(out_shape.NumAxes() - 1, out_shape.At(out_shape.NumAxes() - 1) * (8 / num_bits));
  const int64_t group_dim_size = out_shape.At(group_dim);
  CHECK_GE_OR_RETURN(group_size, 0);
  CHECK_EQ_OR_RETURN(group_dim_size % group_size, 0);
  const int64_t num_groups = group_dim_size / group_size;
  CHECK_EQ_OR_RETURN(scale_shape.NumAxes(), in_shape.NumAxes());
  if (ctx->has_input("zero", 0)) {
    CHECK_EQ_OR_RETURN(ctx->InputShape("zero", 0).NumAxes(), in_shape.NumAxes());
  }
  for (int64_t i = 0; i < out_shape.NumAxes(); ++i) {
    if (i == group_dim) {
      CHECK_EQ_OR_RETURN(scale_shape.At(i), num_groups);
      if (ctx->has_input("zero", 0)) {
        CHECK_EQ_OR_RETURN(ctx->InputShape("zero", 0).At(i), num_groups);
      }
    } else {
      CHECK_EQ_OR_RETURN(scale_shape.At(i), out_shape.At(i));
      if (ctx->has_input("zero", 0)) {
        CHECK_EQ_OR_RETURN(ctx->InputShape("zero", 0).At(i), out_shape.At(i));
      }
    }
  }
  ctx->SetOutputShape("out", 0, out_shape);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> GroupwiseDequantizeOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/*static*/ Maybe<void> GroupwiseDequantizeOp::InferDataType(user_op::InferContext* ctx) {
  const DataType data_type = ctx->InputDType("scale", 0);
  if (ctx->has_input("zero", 0)) { CHECK_EQ_OR_RETURN(ctx->InputDType("zero", 0), data_type); }
  if (ctx->Attr<bool>("symmetric")) {
    CHECK_OR_RETURN(ctx->InputDType("in", 0) == DataType::kUInt8
                    || ctx->InputDType("in", 0) == DataType::kInt8);
  } else {
    CHECK_EQ_OR_RETURN(ctx->InputDType("in", 0), DataType::kUInt8);
  }
  ctx->SetOutputDType("out", 0, data_type);
  return Maybe<void>::Ok();
}

}  // namespace oneflow
