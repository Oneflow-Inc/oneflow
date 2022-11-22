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

Maybe<void> InferTensorDesc4Matmul(user_op::InferContext* ctx) {
  bool transpose_a = ctx->Attr<bool>("transpose_a");
  bool transpose_b = ctx->Attr<bool>("transpose_b");

  const user_op::TensorDesc& a = ctx->InputTensorDesc("a", 0);
  const user_op::TensorDesc& b = ctx->InputTensorDesc("b", 0);
  CHECK_EQ_OR_RETURN(a.shape().NumAxes(), b.shape().NumAxes());
  CHECK_GE_OR_RETURN(a.shape().NumAxes(), 2);
  size_t num_axes = a.shape().NumAxes();

  if (num_axes > 2) {
    for (int i = 0; i < num_axes - 2; ++i) { CHECK_EQ_OR_RETURN(a.shape().At(i), b.shape().At(i)); }
  }

  user_op::TensorDesc* out = ctx->MutOutputTensorDesc("out", 0);

  Shape output = ctx->InputShape("a", 0);
  ctx->SetOutputIsDynamic("out", 0, ctx->InputIsDynamic("a", 0));

  int64_t m, n, k;  // tensor a (no trans): m*k, tensor b (no trans): k*n
  if (!transpose_a) {
    m = a.shape().At(num_axes - 2);
    k = a.shape().At(num_axes - 1);
  } else {
    m = a.shape().At(num_axes - 1);
    k = a.shape().At(num_axes - 2);
  }
  if (!transpose_b) {
    CHECK_EQ_OR_RETURN(k, b.shape().At(num_axes - 2));
    n = b.shape().At(num_axes - 1);
  } else {
    CHECK_EQ_OR_RETURN(k, b.shape().At(num_axes - 1));
    n = b.shape().At(num_axes - 2);
  }
  output.Set(num_axes - 2, m);
  output.Set(num_axes - 1, n);
  out->set_shape(output);
  if (ctx->has_input("_add_to_output", 0)) {
    const auto& add_to_output = ctx->InputTensorDesc("_add_to_output", 0);
    CHECK_EQ_OR_RETURN(add_to_output.shape(), out->shape());
  }
  return Maybe<void>::Ok();
}

Maybe<void> InferDataType4Matmul(user_op::InferContext* ctx) {
  DataType dtype = ctx->InputDType("a", 0);
  CHECK_EQ_OR_RETURN(ctx->InputDType("b", 0), dtype)
      << "InferDataType Failed. Expected " << DataType_Name(dtype) << ", but got "
      << DataType_Name(ctx->InputDType("b", 0));
  if (ctx->has_input("_add_to_output", 0)) {
    CHECK_EQ_OR_RETURN(ctx->InputDType("_add_to_output", 0), dtype)
        << "InferDataType Failed. Expected " << DataType_Name(dtype) << ", but got "
        << DataType_Name(ctx->InputDType("_add_to_output", 0));
  }
  ctx->SetOutputDType("out", 0, dtype);
  return Maybe<void>::Ok();
}

// Theoretically computation cost of matrix multiplication is the products of the number of matrix
// and first dimension of matrix a, second dimension of matrix a, second dimension of matrix
// b. If there is any splitting sbp parallel, the computation cost will be divided by number of
// machines. If we use S(1) at matrix a and S(0) at matrix b, then it will be P at output matrix.
// This is why we don't use SbpParallel at output matrix.
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

/* static */ Maybe<void> MatmulOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return InferTensorDesc4Matmul(ctx);
}

/*static*/ Maybe<void> MatmulOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/*static*/ Maybe<double> MatmulOp::GetComputeComplexity(user_op::ComputeComplexityFnContext* ctx) {
  return GetComputationCost(ctx);
}

/* static */ Maybe<void> MatmulOp::GetSbp(user_op::SbpContext* ctx) {
  // (m, k_a) * (k_b, n) where k_a == k_b
  int32_t m_axis = -1;
  int32_t k_a_axis = -1;
  int32_t k_b_axis = -1;
  int32_t n_axis = -1;
  if (ctx->Attr<bool>("transpose_a")) {
    m_axis = 1;
    k_a_axis = 0;
  } else {
    m_axis = 0;
    k_a_axis = 1;
  }
  if (ctx->Attr<bool>("transpose_b")) {
    k_b_axis = 1;
    n_axis = 0;
  } else {
    k_b_axis = 0;
    n_axis = 1;
  }
  std::vector<user_op::OpArg> out_and_add_to_output_args;
  out_and_add_to_output_args.emplace_back("out", 0);
  if (ctx->user_op_conf().has_input("_add_to_output", 0)) {
    out_and_add_to_output_args.emplace_back("_add_to_output", 0);
  }
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
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> MatmulOp::InferDataType(user_op::InferContext* ctx) {
  return InferDataType4Matmul(ctx);
}

// BatchMatmul

/* static */ Maybe<void> BatchMatmulOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return InferTensorDesc4Matmul(ctx);
}

/*static*/ Maybe<void> BatchMatmulOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> BatchMatmulOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& a_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("a", 0);
  std::vector<user_op::OpArg> out_and_add_to_output_args;
  out_and_add_to_output_args.emplace_back("out", 0);
  if (ctx->user_op_conf().has_input("_add_to_output", 0)) {
    out_and_add_to_output_args.emplace_back("_add_to_output", 0);
  }
  int32_t num_axes = a_tensor.shape().NumAxes();
  FOR_RANGE(int64_t, i, 0, num_axes - 2) {
    ctx->NewBuilder().Split(ctx->inputs(), i).Split(out_and_add_to_output_args, i).Build();
  }
  int32_t m_axis = -1;
  int32_t k_a_axis = -1;
  int32_t k_b_axis = -1;
  int32_t n_axis = -1;
  if (ctx->Attr<bool>("transpose_a")) {
    m_axis = num_axes - 1;
    k_a_axis = num_axes - 2;
  } else {
    m_axis = num_axes - 2;
    k_a_axis = num_axes - 1;
  }
  if (ctx->Attr<bool>("transpose_b")) {
    k_b_axis = num_axes - 1;
    n_axis = num_axes - 2;
  } else {
    k_b_axis = num_axes - 2;
    n_axis = num_axes - 1;
  }
  ctx->NewBuilder()
      .Split(user_op::OpArg("a", 0), m_axis)
      .Broadcast(user_op::OpArg("b", 0))
      .Split(out_and_add_to_output_args, num_axes - 2)
      .Build();
  ctx->NewBuilder()
      .Broadcast(user_op::OpArg("a", 0))
      .Split(user_op::OpArg("b", 0), n_axis)
      .Split(out_and_add_to_output_args, num_axes - 1)
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
  return Maybe<void>::Ok();
}

/*static*/ Maybe<double> BatchMatmulOp::GetComputeComplexity(
    user_op::ComputeComplexityFnContext* ctx) {
  return GetComputationCost(ctx);
}

/* static */ Maybe<void> BatchMatmulOp::InferDataType(user_op::InferContext* ctx) {
  return InferDataType4Matmul(ctx);
}

// BroadcastMatmul

/* static */ Maybe<void> BroadcastMatmulOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  bool transpose_a = ctx->Attr<bool>("transpose_a");
  bool transpose_b = ctx->Attr<bool>("transpose_b");

  const user_op::TensorDesc& a = ctx->InputTensorDesc("a", 0);
  const user_op::TensorDesc& b = ctx->InputTensorDesc("b", 0);
  user_op::TensorDesc* out = ctx->MutOutputTensorDesc("out", 0);

  const int64_t num_a_dims = a.shape().NumAxes();
  const int64_t num_b_dims = b.shape().NumAxes();
  const size_t num_max_batch_dims = std::max(num_a_dims, num_b_dims) - 2;
  auto MakeGetBatchDim = [num_max_batch_dims](size_t num_dims, const Shape& shape_dim) {
    const int64_t num_batch_dims = num_dims - 2;
    const int64_t num_padding_dims = num_max_batch_dims - num_batch_dims;
    return [num_padding_dims, shape_dim](size_t index) {
      return index < num_padding_dims ? 1 : shape_dim.At(index - num_padding_dims);
    };
  };
  auto GetABatchDim = MakeGetBatchDim(num_a_dims, a.shape());
  auto GetBBatchDim = MakeGetBatchDim(num_b_dims, b.shape());

  DimVector out_dim_vec(std::max(num_a_dims, num_b_dims));
  FOR_RANGE(int64_t, i, 0, out_dim_vec.size() - 2) {
    // Set broadcast shape
    //                       m  k          k  n
    // For example: A(16, 1, 4, 8) B(1, 8, 8, 6)
    // We First set the previous batch dims to broadcasted shape: C(16, 8)
    // Then we emplace back m, n -> C(16, 8, 4, 6)
    const int64_t a_batch_dim = GetABatchDim(i);
    const int64_t b_batch_dim = GetBBatchDim(i);
    CHECK(((a_batch_dim != 1 && b_batch_dim == 1) || (a_batch_dim == 1 && b_batch_dim != 1)
           || (a_batch_dim == b_batch_dim)))
        << "Batch Dims could not broadcast, please check. ";
    out_dim_vec[i] = std::max(a_batch_dim, b_batch_dim);
  }
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;  // tensor a (no trans): batch_dims*m*k, tensor b (no trans): batch_dims*k*n
  if (!transpose_a) {
    m = a.shape().At(num_a_dims - 2);
    k = a.shape().At(num_a_dims - 1);
  } else {
    m = a.shape().At(num_a_dims - 1);
    k = a.shape().At(num_a_dims - 2);
  }
  if (!transpose_b) {
    CHECK_EQ_OR_RETURN(k, b.shape().At(num_b_dims - 2))
        << "K dim should be equal to b.shape().At(num_b_dims - 2). ";
    n = b.shape().At(num_b_dims - 1);
  } else {
    CHECK_EQ_OR_RETURN(k, b.shape().At(num_b_dims - 1))
        << "K dim should be equal to b.shape().At(num_b_dims - 1). ";
    n = b.shape().At(num_b_dims - 2);
  }
  out_dim_vec.at(num_max_batch_dims) = m;
  out_dim_vec.at(num_max_batch_dims + 1) = n;
  out->set_shape(Shape(out_dim_vec));

  if (ctx->has_input("_add_to_output", 0)) {
    const user_op::TensorDesc& add_to_output = ctx->InputTensorDesc("_add_to_output", 0);
    CHECK_EQ_OR_RETURN(add_to_output.shape(), out->shape());
  }

  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> BroadcastMatmulOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> BroadcastMatmulOp::GetSbp(user_op::SbpContext* ctx) {
  // (b, m, k) * (k, n) when transpose_b is false
  // (b, m, k) * (n, k) when transpose_b is true
  bool transpose_a = ctx->Attr<bool>("transpose_a");
  bool transpose_b = ctx->Attr<bool>("transpose_b");

  const auto& a_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("a", 0).shape();
  const auto& b_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("b", 0).shape();

  const int64_t a_num_axes = a_shape.NumAxes();
  const int64_t b_num_axes = b_shape.NumAxes();

  int32_t m_a_axis = -1;
  int32_t k_a_axis = -1;
  int32_t k_b_axis = -1;
  int32_t n_axis = -1;

  if (transpose_a) {
    m_a_axis = a_num_axes - 1;
    k_a_axis = a_num_axes - 2;
  } else {
    m_a_axis = a_num_axes - 2;
    k_a_axis = a_num_axes - 1;
  }
  if (transpose_b) {
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

  const int64_t a_batch_dims = a_num_axes - 2;
  const int64_t b_batch_dims = b_num_axes - 2;
  const int64_t max_num_axes = std::max(a_num_axes, b_num_axes);
  const size_t num_max_batch_dims = max_num_axes - 2;
  auto MakeGetBatchDim = [num_max_batch_dims](size_t num_dims, const Shape& shape_dim) {
    const int64_t num_batch_dims = num_dims - 2;
    const int64_t num_padding_dims = num_max_batch_dims - num_batch_dims;
    return [num_padding_dims, shape_dim](size_t index) {
      return index < num_padding_dims ? 1 : shape_dim.At(index - num_padding_dims);
    };
  };
  auto GetABatchDim = MakeGetBatchDim(a_num_axes, a_shape);
  auto GetBBatchDim = MakeGetBatchDim(b_num_axes, b_shape);

  for (int i = 0; i < num_max_batch_dims; i++) {
    const int64_t a_batch_dim = GetABatchDim(i);
    const int64_t b_batch_dim = GetBBatchDim(i);

    if (a_batch_dim == b_batch_dim && a_batch_dim != 1) {
      // S(b axis) x S(b axis) -> S(b axis)
      ctx->NewBuilder()
          .Split(user_op::OpArg("a", 0), i - (num_max_batch_dims - a_batch_dims))
          .Split(user_op::OpArg("b", 0), i - (num_max_batch_dims - b_batch_dims))
          .Split(out_and_add_to_output_args, i)
          .Build();
    } else if (a_batch_dim == 1 && b_batch_dim != 1) {
      // B x S(b axis) -> S(b axis)
      ctx->NewBuilder()
          .Broadcast(user_op::OpArg("a", 0))
          .Split(user_op::OpArg("b", 0), i - (num_max_batch_dims - b_batch_dims))
          .Split(out_and_add_to_output_args, i)
          .Build();
    } else if (b_batch_dim == 1 && a_batch_dim != 1) {
      // S(b axis) x B -> S(b axis)
      ctx->NewBuilder()
          .Split(user_op::OpArg("a", 0), i - (num_max_batch_dims - a_batch_dims))
          .Broadcast(user_op::OpArg("b", 0))
          .Split(out_and_add_to_output_args, i)
          .Build();
    }
  }

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
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> BroadcastMatmulOp::InferDataType(user_op::InferContext* ctx) {
  return InferDataType4Matmul(ctx);
}

/*static*/ Maybe<double> BroadcastMatmulOp::GetComputeComplexity(
    user_op::ComputeComplexityFnContext* ctx) {
  return GetComputationCost(ctx);
}

/* static */ Maybe<void> BroadcastMatmulGradBOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  const user_op::TensorDesc& a = ctx->InputTensorDesc("a", 0);
  const user_op::TensorDesc& b = ctx->InputTensorDesc("b", 0);
  user_op::TensorDesc* out = ctx->MutOutputTensorDesc("out", 0);

  CHECK_EQ_OR_RETURN(a.shape().NumAxes(), b.shape().NumAxes());
  for (int i = 0; i < a.shape().NumAxes() - 1; ++i) {
    CHECK_EQ_OR_RETURN(a.shape().At(i), b.shape().At(i));
  }
  out->set_shape(
      Shape({a.shape().At(a.shape().NumAxes() - 1), b.shape().At(b.shape().NumAxes() - 1)}));

  if (ctx->has_input("_add_to_output", 0)) {
    const user_op::TensorDesc& add_to_output = ctx->InputTensorDesc("_add_to_output", 0);
    CHECK_EQ_OR_RETURN(add_to_output.shape(), out->shape());
  }

  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> BroadcastMatmulGradBOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/*static*/ Maybe<double> BroadcastMatmulGradBOp::GetComputeComplexity(
    user_op::ComputeComplexityFnContext* ctx) {
  const Shape& shape_a = ctx->Shape4ArgNameAndIndex("a", 0);
  int64_t n = shape_a.At(shape_a.NumAxes() - 2);

  double logical_computation_cost = 2 * ctx->Shape4ArgNameAndIndex("b", 0).elem_cnt() * n;
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
/* static */ Maybe<void> BroadcastMatmulGradBOp::GetSbp(user_op::SbpContext* ctx) {
  const auto& a_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("a", 0).shape();
  int64_t last_axis = a_shape.NumAxes() - 1;
  std::vector<user_op::OpArg> out_and_add_to_output_args;
  out_and_add_to_output_args.emplace_back("out", 0);
  if (ctx->user_op_conf().has_input("_add_to_output", 0)) {
    out_and_add_to_output_args.emplace_back("_add_to_output", 0);
  }
  // S(b or m axis) x S(b or m axis) -> P
  for (int64_t i = 0; i < last_axis; ++i) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("a", 0), i)
        .Split(user_op::OpArg("b", 0), i)
        .PartialSum(out_and_add_to_output_args)
        .Build();
  }
  // (b, m, k) * (b, m, n) -> (k, n) [transpose a]
  // S(k) x B -> S(0) or B x S(n) -> S(1)
  // (b, m, n) * (b, m, k) -> (n, k) [transpose a]
  // S(n) x B -> S(0) or B x S(k) -> S(1)
  ctx->NewBuilder()
      .Split(user_op::OpArg("a", 0), last_axis)
      .Broadcast(user_op::OpArg("b", 0))
      .Split(out_and_add_to_output_args, 0)
      .Build();
  ctx->NewBuilder()
      .Broadcast(user_op::OpArg("a", 0))
      .Split(user_op::OpArg("b", 0), last_axis)
      .Split(out_and_add_to_output_args, 1)
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> BroadcastMatmulGradBOp::InferDataType(user_op::InferContext* ctx) {
  return InferDataType4Matmul(ctx);
}

}  // namespace oneflow
