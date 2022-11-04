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
#include <cstdint>
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/op_generated.h"
#include "oneflow/core/framework/user_op_conf.h"

namespace oneflow {

/*static*/ auto FusedRowAttentionWithPairBiasOp::InferDataType(user_op::InferContext* ctx)
    -> Maybe<void> {
  DataType query_type = ctx->InputDType("qmk", 0);
  DataType mask_bias_type = ctx->InputDType("mask_bias", 0);
  DataType pair_bias_type = ctx->InputDType("pair_bias", 0);
  CHECK_EQ_OR_RETURN(mask_bias_type, query_type);
  CHECK_EQ_OR_RETURN(pair_bias_type, query_type);
  ctx->SetOutputDType("out", 0, query_type);
  return Maybe<void>::Ok();
}

/*static*/ auto FusedRowAttentionWithPairBiasOp::InferLogicalTensorDesc(user_op::InferContext* ctx)
    -> Maybe<void> {
  const Shape& qmk_shape = ctx->InputShape("qmk", 0);
  CHECK_EQ_OR_RETURN(qmk_shape.NumAxes(), 4);
  const int64_t batch_size = qmk_shape.At(0);
  const int64_t num_heads = qmk_shape.At(1);
  const int64_t query_lens = qmk_shape.At(2);
  const int64_t key_lens = qmk_shape.At(3);
  CHECK_GT_OR_RETURN(query_lens, 0);
  CHECK_EQ_OR_RETURN(query_lens, key_lens);

  const float scale = ctx->Attr<float>("scale");
  CHECK_LE_OR_RETURN(scale, 1.);

  const Shape& mask_bias_shape = ctx->InputShape("mask_bias", 0);
  CHECK_EQ_OR_RETURN(mask_bias_shape.At(0), batch_size);
  CHECK_EQ_OR_RETURN(mask_bias_shape.At(1), 1);
  CHECK_EQ_OR_RETURN(mask_bias_shape.At(2), 1);
  CHECK_EQ_OR_RETURN(mask_bias_shape.At(3), query_lens);

  const Shape& pair_bias_shape = ctx->InputShape("pair_bias", 0);
  CHECK_EQ_OR_RETURN(pair_bias_shape.At(0), 1);
  CHECK_EQ_OR_RETURN(pair_bias_shape.At(1), num_heads);
  CHECK_EQ_OR_RETURN(pair_bias_shape.At(2), query_lens);
  CHECK_EQ_OR_RETURN(pair_bias_shape.At(3), key_lens);

  ctx->SetOutputShape("out", 0, Shape({batch_size, num_heads, query_lens, key_lens}));
  return Maybe<void>::Ok();
}

/*static*/ auto FusedRowAttentionWithPairBiasOp::InferPhysicalTensorDesc(user_op::InferContext* ctx)
    -> Maybe<void> {
  return InferLogicalTensorDesc(ctx);
}

/*static*/ auto FusedRowAttentionWithPairBiasOp::GetSbp(user_op::SbpContext* ctx) -> Maybe<void> {
  ctx->NewBuilder()
      .Split(user_op::OpArg("qmk", 0), 0)
      .Split(user_op::OpArg("mask_bias", 0), 0)
      .Broadcast(user_op::OpArg("pair_bias", 0))
      .Split(user_op::OpArg("out", 0), 0)
      .Build();
  return Maybe<void>::Ok();
}

}  // namespace oneflow
