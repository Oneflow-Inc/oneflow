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

namespace {

Maybe<double> GetComputationCost(user_op::ComputeComplexityFnContext* ctx) {
  bool transpose_b = ctx->Attr<bool>("transpose_b");
  const Shape& shape_b = ctx->Shape4ArgNameAndIndex("b", 0);
  int64_t n = 0;
  if (!transpose_b) {
    n = shape_b.At(shape_b.NumAxes() - 1);
  } else {
    n = shape_b.At(shape_b.NumAxes() - 2);
  }

  double logical_computation_cost = 2 * ctx->Shape4ArgNameAndIndex("a", 0).elem_cnt() * n;
  const auto& nd_sbp_a = ctx->NdSbp4ArgNameAndIndex("a", 0);
  const auto& nd_sbp_b = ctx->NdSbp4ArgNameAndIndex("b", 0);
  const auto& parallel_hierarchy = ctx->parallel_desc().hierarchy();
  for (int32_t sbp_dim = 0; sbp_dim < nd_sbp_a.sbp_parallel_size(); sbp_dim++) {
    if (nd_sbp_a.sbp_parallel(sbp_dim).has_split_parallel()
        || nd_sbp_b.sbp_parallel(sbp_dim).has_split_parallel()) {
      logical_computation_cost /= parallel_hierarchy->At(sbp_dim);
    }
  }
  return logical_computation_cost;
}

}  // namespace

// BroadcastMatmul

/* static */ Maybe<void> MatmulQuantOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  bool transpose_a = ctx->Attr<bool>("transpose_a");
  bool transpose_b = ctx->Attr<bool>("transpose_b");
  CHECK_OR_RETURN(transpose_b);

  const user_op::TensorDesc& a = ctx->InputTensorDesc("a", 0);
  const user_op::TensorDesc& b = ctx->InputTensorDesc("b", 0);
  // CHECK_EQ_OR_RETURN(a.shape().NumAxes(), b.shape().NumAxes());
  CHECK_GE_OR_RETURN(a.shape().NumAxes(), 2);
  CHECK_EQ_OR_RETURN(b.shape().NumAxes(), 2);
  size_t a_num_axes = a.shape().NumAxes();
  size_t b_num_axes = b.shape().NumAxes();

  user_op::TensorDesc* out = ctx->MutOutputTensorDesc("out", 0);

  Shape output = ctx->InputShape("a", 0);
  ctx->SetOutputIsDynamic("out", 0, ctx->InputIsDynamic("a", 0));

  int64_t m, n, k;  // tensor a (no trans): m*k, tensor b (no trans): k*n
  if (!transpose_a) {
    m = a.shape().At(a_num_axes - 2);
    k = a.shape().At(a_num_axes - 1);
  } else {
    m = a.shape().At(a_num_axes - 1);
    k = a.shape().At(a_num_axes - 2);
  }
  if (!transpose_b) {
    CHECK_EQ_OR_RETURN(k, b.shape().At(b_num_axes - 2));
    n = b.shape().At(b_num_axes - 1);
  } else {
    CHECK_EQ_OR_RETURN(k, b.shape().At(b_num_axes - 1));
    n = b.shape().At(b_num_axes - 2);
  }
  output.Set(a_num_axes - 2, m);
  output.Set(a_num_axes - 1, n);
  out->set_shape(output);
  if (ctx->has_input("_add_to_output", 0)) {
    const auto& add_to_output = ctx->InputTensorDesc("_add_to_output", 0);
    CHECK_EQ_OR_RETURN(add_to_output.shape(), out->shape());
  }
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> MatmulQuantOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> MatmulQuantOp::GetSbp(user_op::SbpContext* ctx) {
  // (m, k_a) * (k_b, n) where k_a == k_b
  const auto& a_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("a", 0).shape();
  const auto& b_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("b", 0).shape();
  const int64_t a_num_axes = a_shape.NumAxes();
  const int64_t b_num_axes = b_shape.NumAxes();

  int32_t m_axis = -1;
  int32_t k_a_axis = -1;
  int32_t k_b_axis = -1;
  int32_t n_axis = -1;
  if (ctx->Attr<bool>("transpose_a")) {
    m_axis = a_num_axes - 1;
    k_a_axis = a_num_axes - 2;
  } else {
    m_axis = a_num_axes - 2;
    k_a_axis = a_num_axes - 1;
  }
  if (ctx->Attr<bool>("transpose_b")) {
    k_b_axis = b_num_axes - 1;
    n_axis = b_num_axes - 2;
  } else {
    k_b_axis = b_num_axes - 2;
    n_axis = b_num_axes - 1;
  }
  std::vector<user_op::OpArg> out_and_add_to_output_args;
  out_and_add_to_output_args.emplace_back("out", 0);
  if (ctx->user_op_conf().has_input("_add_to_output", 0)) {
    out_and_add_to_output_args.emplace_back("_add_to_output", 0);
  }
  if (ctx->user_op_conf().has_input("scale", 0)) {
    CHECK_OR_RETURN(ctx->user_op_conf().has_input("bias", 0));
    ctx->NewBuilder()
        .Split(user_op::OpArg("a", 0), m_axis)
        .Broadcast(user_op::OpArg("b", 0))
        .Broadcast(user_op::OpArg("scale", 0))
        .Broadcast(user_op::OpArg("bias", 0))
        .Split(out_and_add_to_output_args, 0)
        .Build();
    ctx->NewBuilder()
        .Broadcast(user_op::OpArg("a", 0))
        .Split(user_op::OpArg("b", 0), n_axis)
        .Split(user_op::OpArg("scale", 0), 0)
        .Split(user_op::OpArg("bias", 0), 0)
        .Split(out_and_add_to_output_args, 1)
        .Build();
    ctx->NewBuilder()
        .Split(user_op::OpArg("a", 0), k_a_axis)
        .Split(user_op::OpArg("b", 0), k_b_axis)
        .Broadcast(user_op::OpArg("scale", 0))
        .Broadcast(user_op::OpArg("bias", 0))
        .PartialSum(out_and_add_to_output_args)
        .Build();
    ctx->NewBuilder()
        .PartialSum(user_op::OpArg("a", 0))
        .Broadcast(user_op::OpArg("b", 0))
        .Broadcast(user_op::OpArg("scale", 0))
        .Broadcast(user_op::OpArg("bias", 0))
        .PartialSum(out_and_add_to_output_args)
        .Build();
    ctx->NewBuilder()
        .Broadcast(user_op::OpArg("a", 0))
        .PartialSum(user_op::OpArg("b", 0))
        .Broadcast(user_op::OpArg("scale", 0))
        .Broadcast(user_op::OpArg("bias", 0))
        .PartialSum(out_and_add_to_output_args)
        .Build();
  } else {
    ctx->NewBuilder()
        .Split(user_op::OpArg("a", 0), m_axis)
        .Broadcast(user_op::OpArg("b", 0))
        .Split(out_and_add_to_output_args, 0)
        .Build();
    ctx->NewBuilder()
        .Broadcast(user_op::OpArg("a", 0))
        .Split(user_op::OpArg("b", 0), n_axis)
        .Split(out_and_add_to_output_args, 1)
        .Build();
    ctx->NewBuilder()
        .Split(user_op::OpArg("a", 0), k_a_axis)
        .Split(user_op::OpArg("b", 0), k_b_axis)
        .PartialSum(out_and_add_to_output_args)
        .Build();
    ctx->NewBuilder()
        .PartialSum(user_op::OpArg("a", 0))
        .Broadcast(user_op::OpArg("b", 0))
        .PartialSum(out_and_add_to_output_args)
        .Build();
    ctx->NewBuilder()
        .Broadcast(user_op::OpArg("a", 0))
        .PartialSum(user_op::OpArg("b", 0))
        .PartialSum(out_and_add_to_output_args)
        .Build();
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> MatmulQuantOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, ctx->Attr<DataType>("out_dtype"));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<double> MatmulQuantOp::GetComputeComplexity(
    user_op::ComputeComplexityFnContext* ctx) {
  return GetComputationCost(ctx);
}

}  // namespace oneflow
