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

static const int kAlignment = 16;

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
  bool transpose_b = ctx->Attr<bool>("transpose_b");
  CHECK_OR_RETURN(transpose_b);

  const user_op::TensorDesc& a = ctx->InputTensorDesc("a", 0);
  const user_op::TensorDesc& b = ctx->InputTensorDesc("b", 0);
  user_op::TensorDesc* out = ctx->MutOutputTensorDesc("out", 0);

  const int64_t num_a_dims = a.shape().NumAxes();
  const int64_t num_b_dims = b.shape().NumAxes();
  CHECK_OR_RETURN(num_a_dims == 2 || num_a_dims == 3);
  CHECK_EQ_OR_RETURN(num_b_dims, 2);
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;  // tensor a (no trans): batch_dims*m*k, tensor b (no trans): batch_dims*k*n
  m = a.shape().At(num_a_dims - 2);
  k = a.shape().At(num_a_dims - 1);
  if (!transpose_b) {
    CHECK_EQ_OR_RETURN(k, b.shape().At(0)) << "K dim should be equal to b.shape().At(0). ";
    n = b.shape().At(1);
  } else {
    CHECK_EQ_OR_RETURN(k, b.shape().At(1)) << "K dim should be equal to b.shape().At(1). ";
    n = b.shape().At(0);
  }

  CHECK_EQ_OR_RETURN(k % kAlignment, 0);
  CHECK_EQ_OR_RETURN(n % kAlignment, 0);

  Shape output = ctx->InputShape("a", 0);
  output.Set(num_a_dims - 2, m);
  output.Set(num_a_dims - 1, n);
  out->set_shape(Shape(output));

  if (ctx->has_input("_add_to_output", 0)) {
    const user_op::TensorDesc& add_to_output = ctx->InputTensorDesc("_add_to_output", 0);
    CHECK_EQ_OR_RETURN(add_to_output.shape(), out->shape());
  }
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> MatmulQuantOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> MatmulQuantOp::GetSbp(user_op::SbpContext* ctx) {
  // (b, m, k) * (k, n) when transpose_b is false
  // (b, m, k) * (n, k) when transpose_b is true
  bool transpose_b = ctx->Attr<bool>("transpose_b");

  const auto& a_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("a", 0).shape();
  const auto& b_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("b", 0).shape();

  const int64_t a_num_axes = a_shape.NumAxes();
  const int64_t b_num_axes = b_shape.NumAxes();

  int32_t m_a_axis = a_num_axes - 2;
  int32_t k_a_axis = a_num_axes - 1;
  int32_t k_b_axis = -1;
  int32_t n_axis = -1;

  if (transpose_b) {
    k_b_axis = b_num_axes - 1;
    n_axis = b_num_axes - 2;
  } else {
    k_b_axis = b_num_axes - 2;
    n_axis = b_num_axes - 1;
  }

  bool has_bias = false;
  for (const auto& pair : ctx->inputs()) {
    if (pair.first == "bias") {
      CHECK_EQ_OR_RETURN(0, pair.second);
      has_bias = true;
      break;
    }
  }
  std::vector<user_op::OpArg> out_and_add_to_output_args;
  out_and_add_to_output_args.emplace_back("out", 0);

  if (ctx->user_op_conf().has_input("_add_to_output", 0)) {
    out_and_add_to_output_args.emplace_back("_add_to_output", 0);
  }

  const int64_t max_num_axes = std::max(a_num_axes, b_num_axes);

  if (has_bias) {
    // S(m axis) x B -> S(m axis)
    ctx->NewBuilder()
        .Split(user_op::OpArg("a", 0), m_a_axis)
        .Broadcast(user_op::OpArg("b", 0))
        .Broadcast(user_op::OpArg("scale", 0))
        .Broadcast(user_op::OpArg("bias", 0))
        .Split(out_and_add_to_output_args, max_num_axes - 2)
        .Build();
    // B x S(n_axis) -> S(n_axis)
    ctx->NewBuilder()
        .Broadcast(user_op::OpArg("a", 0))
        .Split(user_op::OpArg("b", 0), n_axis)
        .Split(user_op::OpArg("scale", 0), 0)
        .Split(user_op::OpArg("bias", 0), 0)
        .Split(out_and_add_to_output_args, max_num_axes - 1)
        .Build();
    // S(a_k_axis) x S(b_k_axis) -> P
    ctx->NewBuilder()
        .Split(user_op::OpArg("a", 0), k_a_axis)
        .Split(user_op::OpArg("b", 0), k_b_axis)
        .Broadcast(user_op::OpArg("scale", 0))
        .PartialSum(user_op::OpArg("bias", 0))
        .PartialSum(out_and_add_to_output_args)
        .Build();
    // P x B -> P
    ctx->NewBuilder()
        .PartialSum(user_op::OpArg("a", 0))
        .Broadcast(user_op::OpArg("b", 0))
        .Broadcast(user_op::OpArg("scale", 0))
        .PartialSum(user_op::OpArg("bias", 0))
        .PartialSum(out_and_add_to_output_args)
        .Build();
  } else {
    // S(m axis) x B -> S(m axis)
    ctx->NewBuilder()
        .Split(user_op::OpArg("a", 0), m_a_axis)
        .Broadcast(user_op::OpArg("b", 0))
        .Split(out_and_add_to_output_args, max_num_axes - 2)
        .Build();

    // B x S(n_axis) -> S(n_axis)
    ctx->NewBuilder()
        .Broadcast(user_op::OpArg("a", 0))
        .Split(user_op::OpArg("b", 0), n_axis)
        .Split(out_and_add_to_output_args, max_num_axes - 1)
        .Build();

    // S(a_k_axis) x S(b_k_axis) -> P
    ctx->NewBuilder()
        .Split(user_op::OpArg("a", 0), k_a_axis)
        .Split(user_op::OpArg("b", 0), k_b_axis)
        .PartialSum(out_and_add_to_output_args)
        .Build();

    // P x B -> P
    ctx->NewBuilder()
        .PartialSum(user_op::OpArg("a", 0))
        .Broadcast(user_op::OpArg("b", 0))
        .PartialSum(out_and_add_to_output_args)
        .Build();

    // B x P -> P
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
