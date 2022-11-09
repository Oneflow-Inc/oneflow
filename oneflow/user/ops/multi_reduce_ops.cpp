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
#include "oneflow/core/framework/nd_sbp.h"

namespace oneflow {

namespace {

Maybe<void> InferMultiReduceOpShape(user_op::InferContext* ctx) {
  CHECK_GT_OR_RETURN(ctx->input_size("x"), 0) << ctx->op_name() << "must have at least 1 input";
  ctx->SetOutputShape("y", 0, Shape({}));
  return Maybe<void>::Ok();
}

Maybe<void> InferMultiReduceOpDataType(user_op::InferContext* ctx) {
  DataType x_0_dtype = ctx->InputDType("x", 0);
  for (size_t i = 1; i < ctx->input_size("x"); ++i) {
    CHECK_EQ_OR_RETURN(ctx->InputDType("x", i), x_0_dtype)
        << ctx->op_name() << ": the " << i << " th input has the different data type with others";
  }
  ctx->SetOutputDType("y", 0, x_0_dtype);
  return Maybe<void>::Ok();
}

Maybe<void> GetMultiReduceOpSbp(user_op::SbpContext* ctx) {
  const auto& x_0 = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
  int64_t min_num_axes = x_0.shape().NumAxes();
  for (size_t i = 1; i < ctx->user_op_conf().input_size("x"); ++i) {
    const auto& x_i = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", i);
    min_num_axes = std::min(min_num_axes, x_i.shape().NumAxes());
  }
  for (int64_t i = 0; i < min_num_axes; ++i) {
    ctx->NewBuilder().Split(ctx->inputs(), i).PartialSum(user_op::OpArg("y", 0)).Build();
  }
  return Maybe<void>::Ok();
}

Maybe<void> InferLocalMultiReduceOpLogicalShape(user_op::InferContext* ctx) {
  CHECK_GT_OR_RETURN(ctx->input_size("x"), 0) << ctx->op_name() << "must have at least 1 input";
  const NdSbp& any_nd_sbp = ctx->NdSbp4ArgNameAndIndex("x", 0);
  for (int32_t i = 1; i < ctx->input_size("x"); ++i) {
    const NdSbp& input_i_sbp = ctx->NdSbp4ArgNameAndIndex("x", i);
    CHECK_OR_RETURN(input_i_sbp == any_nd_sbp)
        << ctx->op_name() << ": the " << i << " th arg has the different sbp with others, "
        << NdSbpToString(input_i_sbp) << " vs. " << NdSbpToString(any_nd_sbp);
  }
  auto rank_mesh = ctx->parallel_desc().hierarchy();
  CHECK_EQ_OR_RETURN(rank_mesh->NumAxes(), any_nd_sbp.sbp_parallel_size())
      << ctx->op_name() << ": ndim of ranks of " << *JUST(PlacementToString(ctx->parallel_desc()))
      << " is mismatched with the size of sbp " << NdSbpToString(any_nd_sbp);
  int64_t split_num = 1;
  for (int64_t i = 0; i < rank_mesh->NumAxes(); ++i) {
    if (any_nd_sbp.sbp_parallel(i).has_split_parallel()) { split_num *= rank_mesh->At(i); }
  }
  ctx->SetOutputShape("y", 0, Shape({split_num}));
  return Maybe<void>::Ok();
}

Maybe<void> InferLocalMultiReduceOpPhysicalShape(user_op::InferContext* ctx) {
  CHECK_GT_OR_RETURN(ctx->input_size("x"), 0) << ctx->op_name() << "must have at least 1 input";
  ctx->SetOutputShape("y", 0, Shape({1}));
  return Maybe<void>::Ok();
}

Maybe<void> GetLocalMultiReduceOpSbp(user_op::SbpContext* ctx) {
  const auto& x_0 = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
  int64_t min_num_axes = x_0.shape().NumAxes();
  for (size_t i = 1; i < ctx->user_op_conf().input_size("x"); ++i) {
    const auto& x_i = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", i);
    min_num_axes = std::min(min_num_axes, x_i.shape().NumAxes());
  }
  for (int64_t i = 0; i < min_num_axes; ++i) {
    ctx->NewBuilder().Split(ctx->inputs(), i).Split(user_op::OpArg("y", 0), 0).Build();
  }
  return Maybe<void>::Ok();
}

}  // namespace

#define DEFINE_MULTI_REDUCE_OP_METHODS(op)                                 \
  Maybe<void> op##Op::InferLogicalTensorDesc(user_op::InferContext* ctx) { \
    return InferMultiReduceOpShape(ctx);                                   \
  }                                                                        \
  Maybe<void> op##Op::InferDataType(user_op::InferContext* ctx) {          \
    return InferMultiReduceOpDataType(ctx);                                \
  }                                                                        \
  Maybe<void> op##Op::GetSbp(user_op::SbpContext* ctx) { return GetMultiReduceOpSbp(ctx); }

DEFINE_MULTI_REDUCE_OP_METHODS(MultiReduceSumPowAbs)
DEFINE_MULTI_REDUCE_OP_METHODS(MultiReduceMaxAbs)
DEFINE_MULTI_REDUCE_OP_METHODS(MultiReduceMinAbs)
#undef DEFINE_MULTI_REDUCE_OP_METHODS

#define DEFINE_LOCAL_MULTI_REDUCE_OP_METHODS(op)                            \
  Maybe<void> op##Op::InferLogicalTensorDesc(user_op::InferContext* ctx) {  \
    return InferLocalMultiReduceOpLogicalShape(ctx);                        \
  }                                                                         \
  Maybe<void> op##Op::InferPhysicalTensorDesc(user_op::InferContext* ctx) { \
    return InferLocalMultiReduceOpPhysicalShape(ctx);                       \
  }                                                                         \
  Maybe<void> op##Op::InferDataType(user_op::InferContext* ctx) {           \
    return InferMultiReduceOpDataType(ctx);                                 \
  }                                                                         \
  Maybe<void> op##Op::GetSbp(user_op::SbpContext* ctx) { return GetLocalMultiReduceOpSbp(ctx); }

DEFINE_LOCAL_MULTI_REDUCE_OP_METHODS(LocalMultiReduceMaxAbs)
DEFINE_LOCAL_MULTI_REDUCE_OP_METHODS(LocalMultiReduceMinAbs)
#undef DEFINE_LOCAL_MULTI_REDUCE_OP_METHODS

}  // namespace oneflow
