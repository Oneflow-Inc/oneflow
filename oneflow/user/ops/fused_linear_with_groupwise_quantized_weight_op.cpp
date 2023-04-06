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
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/just.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/infer_util.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

namespace {

Maybe<void> InferTensorDesc4FusedMatmulBias(user_op::InferContext* ctx) {
  const user_op::TensorDesc& x_desc = ctx->InputTensorDesc("x", 0);
  CHECK_GE_OR_RETURN(x_desc.shape().NumAxes(), 2);
  const int64_t k = x_desc.shape().At(x_desc.shape().NumAxes() - 1);

  const user_op::TensorDesc& w_desc = ctx->InputTensorDesc("w", 0);
  CHECK_EQ_OR_RETURN(w_desc.shape().NumAxes(), 2);
  const int64_t n = w_desc.shape().At(0);
  const int32_t num_bits = ctx->Attr<int32_t>("num_bits");
  if (num_bits == 8) {
    CHECK_EQ_OR_RETURN(w_desc.shape().At(1), k);
  } else if (num_bits == 4) {
    CHECK_EQ_OR_RETURN(w_desc.shape().At(1) * 2, k);
  } else {
    UNIMPLEMENTED_THEN_RETURN();
  }
  const int64_t group_dim = ctx->Attr<int64_t>("group_dim");
  CHECK_OR_RETURN(group_dim == 0 || group_dim == 1);
  const int64_t group_dim_size = group_dim == 0 ? n : k;
  const int64_t group_size = ctx->Attr<int64_t>("group_size");
  CHECK_GT_OR_RETURN(group_size, 1);
  CHECK_LE_OR_RETURN(group_size, group_dim_size);
  CHECK_EQ_OR_RETURN(group_dim_size % group_size, 0);
  const int64_t num_groups = group_dim_size / group_size;
  const user_op::TensorDesc& w_scale_desc = ctx->InputTensorDesc("w_scale", 0);
  CHECK_EQ_OR_RETURN(w_scale_desc.shape().NumAxes(), 2);
  if (group_dim == 0) {
    CHECK_EQ_OR_RETURN(w_scale_desc.shape().At(0), num_groups);
    CHECK_EQ_OR_RETURN(w_scale_desc.shape().At(1), k);
  } else if (group_dim == 1) {
    CHECK_EQ_OR_RETURN(w_scale_desc.shape().At(0), n);
    CHECK_EQ_OR_RETURN(w_scale_desc.shape().At(1), num_groups);
  } else {
    UNIMPLEMENTED_THEN_RETURN();
  }
  Shape out_shape = x_desc.shape();
  out_shape[x_desc.shape().NumAxes() - 1] = n;

  if (ctx->has_input("b", 0)) {
    const user_op::TensorDesc& b_desc = ctx->InputTensorDesc("b", 0);
    CHECK_EQ_OR_RETURN(b_desc.shape().NumAxes(), 1);
    CHECK_EQ_OR_RETURN(b_desc.shape().At(0), n);
  }

  if (ctx->has_input("w_zero", 0)) {
    const user_op::TensorDesc& w_zero_desc = ctx->InputTensorDesc("w_zero", 0);
    CHECK_OR_RETURN(w_zero_desc.shape() == w_scale_desc.shape());
  }

  ctx->SetOutputShape("out", 0, out_shape);

  return Maybe<void>::Ok();
}

Maybe<void> InferDataType4MatmulBias(user_op::InferContext* ctx) {
  const DataType data_type = ctx->InputDType("x", 0);
  CHECK_EQ_OR_RETURN(ctx->InputDType("w_scale", 0), data_type);
  if (ctx->has_input("w_zero", 0)) { CHECK_EQ_OR_RETURN(ctx->InputDType("w_zero", 0), data_type); }
  if (ctx->has_input("b", 0)) { CHECK_EQ_OR_RETURN(ctx->InputDType("b", 0), data_type); }
  if (ctx->Attr<bool>("symmetric")) {
    CHECK_OR_RETURN(ctx->InputDType("w", 0) == DataType::kUInt8
                    || ctx->InputDType("w", 0) == DataType::kInt8);
  } else {
    CHECK_EQ_OR_RETURN(ctx->InputDType("w", 0), DataType::kUInt8);
  }
  ctx->SetOutputDType("out", 0, data_type);
  return Maybe<void>::Ok();
}

}  // namespace

/* static */ Maybe<void> FusedLinearWithGroupwiseQuantizedWeightOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferTensorDesc4FusedMatmulBias(ctx);
}

/*static*/ Maybe<void> FusedLinearWithGroupwiseQuantizedWeightOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> FusedLinearWithGroupwiseQuantizedWeightOp::GetSbp(
    user_op::SbpContext* ctx) {
  // (b, m, k) * (n, k)
  const auto& x_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0).shape();

  const int64_t x_num_axes = x_shape.NumAxes();

  const int64_t out_num_axes = x_num_axes;
  const int32_t k_x_axis = x_num_axes - 1;

  std::vector<user_op::OpArg> bias_args;
  if (ctx->user_op_conf().has_input("b", 0)) { bias_args.emplace_back("b", 0); }

  std::vector<user_op::OpArg> scale_args;
  scale_args.emplace_back("w_scale", 0);
  if (ctx->user_op_conf().has_input("w_zero", 0)) { scale_args.emplace_back("w_zero", 0); }

  for (int i = 0; i < x_shape.NumAxes() - 1; i++) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("x", 0), i)
        .Broadcast(user_op::OpArg("w", 0))
        .Broadcast(scale_args)
        .Broadcast(bias_args)
        .Split(user_op::OpArg("out", 0), i)
        .Build();
  }

  const int64_t group_dim = ctx->user_op_conf().attr<int64_t>("group_dim");
  const int64_t group_size = ctx->user_op_conf().attr<int64_t>("group_size");
  CHECK_OR_RETURN(group_dim == 0 || group_dim == 1);
  const auto& x_desc = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
  const auto& w_desc = ctx->LogicalTensorDesc4InputArgNameAndIndex("w", 0);
  CHECK_GE_OR_RETURN(x_desc.shape().NumAxes(), 2);
  CHECK_EQ_OR_RETURN(w_desc.shape().NumAxes(), 2);
  const int64_t k = x_desc.shape().At(x_desc.shape().NumAxes() - 1);
  const int64_t n = w_desc.shape().At(0);
  const int64_t group_dim_size = group_dim == 0 ? k : n;
  CHECK_EQ_OR_RETURN(group_dim_size % group_size, 0);
  const int64_t num_groups = group_dim_size / group_size;

  // B x S(n_axis) -> S(n_axis)
  if (group_dim == 1 || num_groups % ctx->parallel_num() == 0) {
    ctx->NewBuilder()
        .Broadcast(user_op::OpArg("x", 0))
        .Split(user_op::OpArg("w", 0), 0)
        .Split(scale_args, 0)
        .Split(bias_args, 0)
        .Split(user_op::OpArg("out", 0), out_num_axes - 1)
        .Build();
  }

  // S(x_k_axis) x S(w_k_axis) -> P
  if (group_dim == 0 || num_groups % ctx->parallel_num() == 0) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("x", 0), k_x_axis)
        .Split(user_op::OpArg("w", 0), 1)
        .Split(scale_args, 1)
        .PartialSum(bias_args)
        .PartialSum(user_op::OpArg("out", 0))
        .Build();
  }

  // P x B -> P
  ctx->NewBuilder()
      .PartialSum(user_op::OpArg("x", 0))
      .Broadcast(user_op::OpArg("w", 0))
      .Broadcast(scale_args)
      .PartialSum(bias_args)
      .PartialSum(user_op::OpArg("out", 0))
      .Build();

  return Maybe<void>::Ok();
}

/* static */ Maybe<void> FusedLinearWithGroupwiseQuantizedWeightOp::InferDataType(
    user_op::InferContext* ctx) {
  return InferDataType4MatmulBias(ctx);
}

}  // namespace oneflow
