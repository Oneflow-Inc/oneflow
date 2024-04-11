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
#include "oneflow/core/common/shape.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

Maybe<void> ScaledDotProductFlashAttentionOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& q_shape = ctx->InputShape("query", 0);
  const Shape& k_shape = ctx->InputShape("key", 0);
  const Shape& v_shape = ctx->InputShape("value", 0);

  auto batch_size = q_shape.At(0);
  auto seqlen_q = q_shape.At(1);
  auto num_heads = q_shape.At(2);
  auto head_size_og = q_shape.At(3);
  auto seqlen_k = k_shape.At(1);
  auto num_heads_k = k_shape.At(2);

  // check input tensor shape.
  CHECK_EQ_OR_RETURN(batch_size, k_shape.At(0)) << "query has different batch size from key.";
  CHECK_EQ_OR_RETURN(batch_size, v_shape.At(0)) << "query has different batch size from value.";

  CHECK_EQ_OR_RETURN(seqlen_k, v_shape.At(1)) << "key has different seqlen from value.";
  CHECK_EQ_OR_RETURN(num_heads_k, v_shape.At(2)) << "key has different num_heads from value.";

  CHECK_EQ_OR_RETURN(head_size_og, k_shape.At(3)) << "query has different head_size from key";
  CHECK_EQ_OR_RETURN(head_size_og, v_shape.At(3)) << "query has different head_size from value";

  // batch size must be positive.
  CHECK_GT_OR_RETURN(batch_size, 0) << "batch size must be positive";

  // only support head dimensions at most 256.
  CHECK_LE_OR_RETURN(head_size_og, 256) << "only support head dimensions at most 256";

  // number of heads in key/value must devide number of heads in query.
  CHECK_EQ_OR_RETURN(num_heads % num_heads_k, 0) << "number of heads in key/value must devide number of heads in query.";

  ctx->SetOutputShape("out", 0, Shape({batch_size, seqlen_q, num_heads, head_size_og}));
  // save for backward
  ctx->SetOutputShape("softmax_lse", 0, Shape({batch_size, num_heads, seqlen_q}));
  // save seed and offset for backward.
  ctx->SetOutputShape("rng_state", 0, Shape({2}));

  return Maybe<void>::Ok();
}

Maybe<void> ScaledDotProductFlashAttentionOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return ScaledDotProductFlashAttentionOp::InferLogicalTensorDesc(ctx);
}

Maybe<void> ScaledDotProductFlashAttentionOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder().Broadcast(user_op::OpArg("query", 0))
                   .Broadcast(user_op::OpArg("key", 0))
                   .Broadcast(user_op::OpArg("value", 0))
                   .Broadcast(user_op::OpArg("out", 0))
                   .Broadcast(user_op::OpArg("softmax", 0))
                   .Build();
  return Maybe<void>::Ok();
}

Maybe<void> ScaledDotProductFlashAttentionOp::InferDataType(user_op::InferContext* ctx) {
  auto q_datatype = ctx->InputDType("query", 0);
  auto k_datatype = ctx->InputDType("key", 0);
  auto v_datatype = ctx->InputDType("value", 0);

  CHECK_EQ_OR_RETURN(q_datatype, k_datatype) << "query has different data type from key.";
  CHECK_EQ_OR_RETURN(q_datatype, v_datatype) << "query has different data type from value.";

  ctx->SetOutputDType("out", 0, q_datatype);
  ctx->SetOutputDType("softmax_lse", 0, DataType::kFloat);
  ctx->SetOutputDType("rng_state", 0, DataType::kUInt64);

  return Maybe<void>::Ok();
}

}  // namespace oneflow