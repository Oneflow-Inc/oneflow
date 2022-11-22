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

Maybe<void> CheckInShape(user_op::InferContext* ctx) {
  const Shape& q_shape = ctx->InputShape("query", 0);
  const Shape& k_shape = ctx->InputShape("key", 0);
  const Shape& v_shape = ctx->InputShape("value", 0);
  const Shape& cu_seqlens_q_shape = ctx->InputShape("cu_seqlens_q", 0);
  const Shape& cu_seqlens_k_shape = ctx->InputShape("cu_seqlens_k", 0);
  CHECK_EQ_OR_RETURN(q_shape.NumAxes(), 4)
      << "query shape num_axes should be 4, but got " << q_shape.NumAxes();
  const int64_t batch_size = q_shape.At(0);
  const int64_t num_head = q_shape.At(2);
  const int64_t head_size = q_shape.At(3);
  CHECK_OR_RETURN(head_size == 16 || head_size == 32 || head_size == 64 || head_size == 128)
      << "flash-attention only support head_size in (16, 32, 64, 128).";
  CHECK_EQ_OR_RETURN(k_shape.NumAxes(), 4)
      << "key shape num_axes should be 4, but got " << k_shape.NumAxes();
  CHECK_EQ_OR_RETURN(v_shape.NumAxes(), 4)
      << "key shape num_axes should be 4, but got " << v_shape.NumAxes();
  CHECK_EQ_OR_RETURN(batch_size, k_shape.At(0));
  CHECK_EQ_OR_RETURN(batch_size, v_shape.At(0));
  CHECK_EQ_OR_RETURN(k_shape.At(1), v_shape.At(1));
  CHECK_EQ_OR_RETURN(num_head, k_shape.At(2));
  CHECK_EQ_OR_RETURN(num_head, v_shape.At(2));
  CHECK_EQ_OR_RETURN(head_size, k_shape.At(3));
  CHECK_EQ_OR_RETURN(head_size, v_shape.At(3));
  CHECK_EQ_OR_RETURN(cu_seqlens_q_shape.At(0), batch_size + 1);
  CHECK_EQ_OR_RETURN(cu_seqlens_k_shape.At(0), batch_size + 1);
  return Maybe<void>::Ok();
}

}  // namespace

/* static */ Maybe<void> FlashAttentionOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  JUST(CheckInShape(ctx));
  const Shape& q_shape = ctx->InputShape("query", 0);
  const int64_t batch_size = q_shape.At(0);
  const int64_t seq_len_q = q_shape.At(1);
  const int64_t num_head = q_shape.At(2);
  ctx->SetOutputShape("out", 0, q_shape);
  ctx->SetOutputShape("softmax_lse", 0, Shape({batch_size, num_head, seq_len_q}));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> FlashAttentionOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> FlashAttentionOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Split(ctx->inputs(), 0)
      // TODO:(guoran), cu_seqlens_q size now is batch_size + 1, can it be batch_size? so can set
      // to s0
      .Broadcast(user_op::OpArg("cu_seqlens_q", 0))
      .Broadcast(user_op::OpArg("cu_seqlens_k", 0))
      .Split(ctx->outputs(), 0)
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> FlashAttentionOp::InferDataType(user_op::InferContext* ctx) {
  const DataType& q_dtype = ctx->InputDType("query", 0);
  const DataType& cu_seqlens_q_dtype = ctx->InputDType("cu_seqlens_q", 0);
  const DataType& cu_seqlens_k_dtype = ctx->InputDType("cu_seqlens_k", 0);
  CHECK_EQ_OR_RETURN(cu_seqlens_q_dtype, DataType::kInt32);
  CHECK_EQ_OR_RETURN(cu_seqlens_k_dtype, DataType::kInt32);
  ctx->SetOutputDType("out", 0, q_dtype);
  DataType softmax_lse_dtype = (q_dtype == DataType::kFloat16 || q_dtype == DataType::kBFloat16)
                                   ? DataType::kFloat
                                   : q_dtype;
  ctx->SetOutputDType("softmax_lse", 0, softmax_lse_dtype);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> FlashAttentionGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  JUST(CheckInShape(ctx));
  const Shape& q_shape = ctx->InputShape("query", 0);
  const Shape& k_shape = ctx->InputShape("key", 0);
  const Shape& v_shape = ctx->InputShape("value", 0);
  ctx->SetOutputShape("query_grad", 0, q_shape);
  ctx->SetOutputShape("key_grad", 0, k_shape);
  ctx->SetOutputShape("value_grad", 0, v_shape);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> FlashAttentionGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> FlashAttentionGradOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Split(ctx->inputs(), 0)
      // TODO:(guoran), cu_seqlens_q size now is batch_size + 1, can it be batch_size? so can set
      // to s0
      .Broadcast(user_op::OpArg("cu_seqlens_q", 0))
      .Broadcast(user_op::OpArg("cu_seqlens_k", 0))
      .Split(ctx->outputs(), 0)
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> FlashAttentionGradOp::InferDataType(user_op::InferContext* ctx) {
  const DataType& cu_seqlens_q_dtype = ctx->InputDType("cu_seqlens_q", 0);
  const DataType& cu_seqlens_k_dtype = ctx->InputDType("cu_seqlens_k", 0);
  CHECK_EQ_OR_RETURN(cu_seqlens_q_dtype, DataType::kInt32);
  CHECK_EQ_OR_RETURN(cu_seqlens_k_dtype, DataType::kInt32);
  ctx->SetOutputDType("query_grad", 0, ctx->InputDType("query", 0));
  ctx->SetOutputDType("key_grad", 0, ctx->InputDType("key", 0));
  ctx->SetOutputDType("value_grad", 0, ctx->InputDType("value", 0));
  return Maybe<void>::Ok();
}

}  // namespace oneflow
