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

  user_op::TensorDesc* out = ctx->OutputTensorDesc("out", 0);

  *ctx->OutputShape("out", 0) = ctx->InputShape("a", 0);
  *ctx->OutputIsDynamic("out", 0) = ctx->InputIsDynamic("a", 0);

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
  out->mut_shape()->Set(num_axes - 2, m);
  out->mut_shape()->Set(num_axes - 1, n);
  if (ctx->has_input("_add_to_output", 0)) {
    const auto& add_to_output = ctx->InputTensorDesc("_add_to_output", 0);
    CHECK_EQ_OR_RETURN(add_to_output.shape(), out->shape());
  }
  return Maybe<void>::Ok();
}

Maybe<void> InferDataType4Matmul(user_op::InferContext* ctx) {
  const DataType& dtype = ctx->InputDType("a", 0);
  CHECK_EQ_OR_RETURN(ctx->InputDType("b", 0), dtype);
  if (ctx->has_input("_add_to_output", 0)) {
    CHECK_EQ_OR_RETURN(ctx->InputDType("_add_to_output", 0), dtype);
  }
  *ctx->OutputDType("out", 0) = dtype;
  return Maybe<void>::Ok();
}

void GenBackwardOpConf4Matmul(const std::string& op_type_name, const user_op::UserOpWrapper& op,
                              user_op::AddOpFn AddOp) {
  const bool transpose_a = op.attr<bool>("transpose_a");
  const bool transpose_b = op.attr<bool>("transpose_b");
  const double alpha = op.attr<double>("alpha");
  auto HandleGradOp = [&](user_op::UserOpConfWrapper&& grad_op,
                          std::string&& input_arg_name) -> void {
    op.BindGradTensorWithOpInput(grad_op.output("out", 0), input_arg_name, 0);
    AddOp(grad_op);
  };

  if (op.NeedGenGradTensor4OpInput("a", 0)) {
    if (transpose_a) {
      user_op::UserOpConfWrapper grad_a_op =
          user_op::UserOpConfWrapperBuilder(op.op_name() + "_grad_a")
              .Op(op_type_name)
              .Input("a", op.input("b", 0))
              .Input("b", op.GetGradTensorWithOpOutput("out", 0))
              .Output("out")
              .Attr<bool>("transpose_a", transpose_b)
              .Attr<bool>("transpose_b", true)
              .Attr<double>("alpha", alpha)
              .Build();
      HandleGradOp(std::move(grad_a_op), "a");
    } else {
      user_op::UserOpConfWrapper grad_a_op =
          user_op::UserOpConfWrapperBuilder(op.op_name() + "_grad_a")
              .Op(op_type_name)
              .Input("a", op.GetGradTensorWithOpOutput("out", 0))
              .Input("b", op.input("b", 0))
              .Output("out")
              .Attr<bool>("transpose_a", false)
              .Attr<bool>("transpose_b", !transpose_b)
              .Attr<double>("alpha", alpha)
              .Build();
      HandleGradOp(std::move(grad_a_op), "a");
    }
  }
  if (op.NeedGenGradTensor4OpInput("b", 0)) {
    if (transpose_b) {
      user_op::UserOpConfWrapper grad_b_op =
          user_op::UserOpConfWrapperBuilder(op.op_name() + "_grad_b")
              .Op(op_type_name)
              .Input("a", op.GetGradTensorWithOpOutput("out", 0))
              .Input("b", op.input("a", 0))
              .Output("out")
              .Attr<bool>("transpose_a", true)
              .Attr<bool>("transpose_b", transpose_a)
              .Attr<double>("alpha", alpha)
              .Build();
      HandleGradOp(std::move(grad_b_op), "b");
    } else {
      user_op::UserOpConfWrapper grad_b_op =
          user_op::UserOpConfWrapperBuilder(op.op_name() + "_grad_b")
              .Op(op_type_name)
              .Input("a", op.input("a", 0))
              .Input("b", op.GetGradTensorWithOpOutput("out", 0))
              .Output("out")
              .Attr<bool>("transpose_a", !transpose_a)
              .Attr<bool>("transpose_b", false)
              .Attr<double>("alpha", alpha)
              .Build();
      HandleGradOp(std::move(grad_b_op), "b");
    }
  }
}

// Theoretically computation cost of matrix multiplication is the products of the number of matrix
// and first dimension of matrix a, second dimension of matrix a, second dimension of matrix
// b. If there is any splitting sbp parallel, the computation cost will be divided by number of
// machines. If we use S(1) at matrix a and S(0) at matrix b, then it will be P at output matrix.
// This is why we don't use SbpParallel at output matrix.
Maybe<double> GetComputationCostFn(user_op::ComputeComplexityFnContext* ctx) {
  bool transpose_b = ctx->Attr<bool>("transpose_b");
  Shape* shape_b = ctx->Shape4ArgNameAndIndex("b", 0);
  int64_t n;
  if (!transpose_b) {
    n = shape_b->At(shape_b->NumAxes() - 1);
  } else {
    n = shape_b->At(shape_b->NumAxes() - 2);
  }

  double logical_computation_cost = 2 * ctx->Shape4ArgNameAndIndex("a", 0)->elem_cnt() * n;
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
  return GetComputationCostFn(ctx);
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
  return GetComputationCostFn(ctx);
}

/* static */ Maybe<void> BatchMatmulOp::InferDataType(user_op::InferContext* ctx) {
  return InferDataType4Matmul(ctx);
}

/* static */ Maybe<void> BroadcastMatmulOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  bool transpose_a = ctx->Attr<bool>("transpose_a");
  bool transpose_b = ctx->Attr<bool>("transpose_b");

  const user_op::TensorDesc& a = ctx->InputTensorDesc("a", 0);
  const user_op::TensorDesc& b = ctx->InputTensorDesc("b", 0);
  user_op::TensorDesc* out = ctx->OutputTensorDesc("out", 0);

  // NOTE: support broadcast b to a for now
  // TODO(zwx): support broadcast a to b
  CHECK_GT_OR_RETURN(a.shape().NumAxes(), b.shape().NumAxes());
  CHECK_EQ_OR_RETURN(b.shape().NumAxes(), 2);
  // NOTE: don't support transpose_a for now
  CHECK_OR_RETURN(!transpose_a);

  DimVector out_dim_vec(a.shape().NumAxes() - 1);
  FOR_RANGE(int64_t, i, 0, out_dim_vec.size()) { out_dim_vec[i] = a.shape().At(i); }
  int64_t k = a.shape().At(a.shape().NumAxes() - 1);
  int64_t n = -1;
  if (!transpose_b) {
    CHECK_EQ_OR_RETURN(k, b.shape().At(b.shape().NumAxes() - 2));
    n = b.shape().At(b.shape().NumAxes() - 1);
  } else {
    CHECK_EQ_OR_RETURN(k, b.shape().At(b.shape().NumAxes() - 1));
    n = b.shape().At(b.shape().NumAxes() - 2);
  }
  out_dim_vec.emplace_back(n);
  *out->mut_shape() = Shape(out_dim_vec);

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
  CHECK_OR_RETURN(!transpose_a);

  const auto& a_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("a", 0).shape();
  int32_t k_a_axis = a_shape.NumAxes() - 1;
  int32_t k_b_axis = -1;
  int32_t n_axis = -1;
  if (transpose_b) {
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

  // S(b or m axis) x B -> S(b or m axis)
  for (int64_t i = 0; i < a_shape.NumAxes() - 1; ++i) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("a", 0), i)
        .Broadcast(user_op::OpArg("b", 0))
        .Split(out_and_add_to_output_args, i)
        .Build();
  }
  // B x S(n_axis) -> S(n_axis)
  ctx->NewBuilder()
      .Broadcast(user_op::OpArg("a", 0))
      .Split(user_op::OpArg("b", 0), n_axis)
      .Split(out_and_add_to_output_args, a_shape.NumAxes() - 1)
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
  return GetComputationCostFn(ctx);
}

/* static */ Maybe<void> BroadcastMatmulGradBOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  const user_op::TensorDesc& a = ctx->InputTensorDesc("a", 0);
  const user_op::TensorDesc& b = ctx->InputTensorDesc("b", 0);
  user_op::TensorDesc* out = ctx->OutputTensorDesc("out", 0);

  CHECK_EQ_OR_RETURN(a.shape().NumAxes(), b.shape().NumAxes());
  for (int i = 0; i < a.shape().NumAxes() - 1; ++i) {
    CHECK_EQ_OR_RETURN(a.shape().At(i), b.shape().At(i));
  }

  *out->mut_shape() =
      Shape({a.shape().At(a.shape().NumAxes() - 1), b.shape().At(b.shape().NumAxes() - 1)});

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
  Shape* shape_a = ctx->Shape4ArgNameAndIndex("a", 0);
  int64_t n = shape_a->At(shape_a->NumAxes() - 2);

  double logical_computation_cost = 2 * ctx->Shape4ArgNameAndIndex("b", 0)->elem_cnt() * n;
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

REGISTER_USER_OP_GRAD("matmul").SetGenBackwardOpConfFn(
    [](const user_op::UserOpWrapper& op, const user_op::AddOpFn& AddOp) -> Maybe<void> {
      GenBackwardOpConf4Matmul("matmul", op, AddOp);
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("batch_matmul")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                               const user_op::AddOpFn& AddOp) -> Maybe<void> {
      GenBackwardOpConf4Matmul("batch_matmul", op, AddOp);
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("broadcast_matmul")
    .SetBackwardOpConfGenFn([](user_op::BackwardOpConfContext* ctx) -> Maybe<void> {
      bool transpose_a = ctx->FwOp().attr<bool>("transpose_a");
      bool transpose_b = ctx->FwOp().attr<bool>("transpose_b");
      double alpha = ctx->FwOp().attr<double>("alpha");
      CHECK_OR_RETURN(!transpose_a);

      std::string a_grad_op_name = ctx->FwOp().op_name() + "_a_grad";
      ctx->DefineOp(a_grad_op_name,
                    [&](user_op::BackwardOpBuilder& builder) -> user_op::UserOpConfWrapper {
                      return builder.OpTypeName("broadcast_matmul")
                          .InputBind("a", ctx->FwOp().output_grad("out", 0))
                          .InputBind("b", ctx->FwOp().input("b", 0))
                          .Attr<bool>("transpose_a", transpose_a)
                          .Attr<bool>("transpose_b", !transpose_b)
                          .Attr<double>("alpha", alpha)
                          .Output("out")
                          .Build();
                    });

      ctx->FwOp().InputGradBind(user_op::OpArg("a", 0), [&]() -> const std::string& {
        return ctx->GetOp(a_grad_op_name).output("out", 0);
      });

      std::string b_grad_op_name = ctx->FwOp().op_name() + "_b_grad";
      ctx->DefineOp(b_grad_op_name,
                    [&](user_op::BackwardOpBuilder& builder) -> user_op::UserOpConfWrapper {
                      if (!transpose_b) {
                        return builder.OpTypeName("broadcast_matmul_grad_b")
                            .InputBind("a", ctx->FwOp().input("a", 0))
                            .InputBind("b", ctx->FwOp().output_grad("out", 0))
                            .Attr<double>("alpha", alpha)
                            .Output("out")
                            .Build();
                      } else {
                        return builder.OpTypeName("broadcast_matmul_grad_b")
                            .InputBind("a", ctx->FwOp().output_grad("out", 0))
                            .InputBind("b", ctx->FwOp().input("a", 0))
                            .Attr<double>("alpha", alpha)
                            .Output("out")
                            .Build();
                      }
                    });

      ctx->FwOp().InputGradBind(user_op::OpArg("b", 0), [&]() -> const std::string& {
        return ctx->GetOp(b_grad_op_name).output("out", 0);
      });
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
