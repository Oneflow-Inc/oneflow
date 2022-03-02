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

/* static */ Maybe<void> IdShuffleOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& ids_shape = ctx->InputShape("ids", 0);
  const int32_t num_columns = ctx->Attr<int32_t>("num_columns");
  if (ctx->has_input("column_ids", 0)) {
    const Shape& column_ids_shape = ctx->InputShape("column_ids", 0);
    CHECK_EQ_OR_RETURN(ids_shape, column_ids_shape);
  } else {
    if (num_columns > 1) {
      CHECK_EQ_OR_RETURN(ids_shape.NumAxes(), 2);
      CHECK_EQ_OR_RETURN(ids_shape.At(1), num_columns);
    }
  }
  const int64_t num_ids = ids_shape.elem_cnt();
  const int64_t parallel_num = ctx->parallel_num();
  *ctx->OutputShape("num_unique_matrix", 0) = Shape({parallel_num * parallel_num});
  *ctx->OutputShape("inverse_unique_partion_indices", 0) = ids_shape;
  *ctx->OutputShape("cur_rank_num_unique", 0) = Shape({1});
  *ctx->OutputShape("cur_rank_unique_ids", 0) = Shape({num_ids * parallel_num});
  *ctx->OutputShape("cur_rank_inverse_indices", 0) = Shape({num_ids * parallel_num});
  *ctx->OutputShape("cur_rank_unique_column_ids", 0) = Shape({num_ids * parallel_num});
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> IdShuffleOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> IdShuffleOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Split(ctx->inputs(), 0)
      .Split(ctx->outputs(), 0)
      .Broadcast(user_op::OpArg("num_unique_matrix", 0))
      .Broadcast(user_op::OpArg("cur_rank_num_unique", 0))
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> IdShuffleOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("num_unique_matrix", 0) = DataType::kUInt32;
  *ctx->OutputDType("inverse_unique_partion_indices", 0) = DataType::kUInt32;
  *ctx->OutputDType("cur_rank_num_unique", 0) = DataType::kUInt32;
  *ctx->OutputDType("cur_rank_unique_ids", 0) = ctx->InputDType("ids", 0);
  *ctx->OutputDType("cur_rank_inverse_indices", 0) = DataType::kUInt32;
  if (ctx->has_input("column_ids", 0)) {
    *ctx->OutputDType("cur_rank_unique_column_ids", 0) = ctx->InputDType("column_ids", 0);
  } else {
    *ctx->OutputDType("cur_rank_unique_column_ids", 0) = DataType::kUInt32;
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> EmbeddingShuffleOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& cur_rank_embeddings_shape = ctx->InputShape("cur_rank_embeddings", 0);
  const Shape& num_unique_matrix_shape = ctx->InputShape("num_unique_matrix", 0);
  const Shape& cur_rank_inverse_indices_shape = ctx->InputShape("cur_rank_inverse_indices", 0);
  const Shape& inverse_unique_partion_indices_shape =
      ctx->InputShape("inverse_unique_partion_indices", 0);
  const int64_t num_ids = inverse_unique_partion_indices_shape.elem_cnt();
  const int64_t parallel_num = ctx->parallel_num();
  CHECK_EQ_OR_RETURN(cur_rank_embeddings_shape.NumAxes(), 2);
  CHECK_EQ_OR_RETURN(cur_rank_embeddings_shape.At(0), parallel_num * num_ids);
  const int64_t embedding_size = cur_rank_embeddings_shape.At(1);
  CHECK_EQ_OR_RETURN(num_unique_matrix_shape.elem_cnt(), parallel_num * parallel_num);
  CHECK_EQ_OR_RETURN(cur_rank_inverse_indices_shape.elem_cnt(), parallel_num * num_ids);
  DimVector out_dim_vec = inverse_unique_partion_indices_shape.dim_vec();
  out_dim_vec.push_back(embedding_size);
  *ctx->OutputShape("embeddings", 0) = Shape(out_dim_vec);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> EmbeddingShuffleOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> EmbeddingShuffleOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Split(ctx->inputs(), 0)
      .Broadcast(user_op::OpArg("num_unique_matrix", 0))
      .Split(ctx->outputs(), 0)
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> EmbeddingShuffleOp::InferDataType(user_op::InferContext* ctx) {
  CHECK_OR_RETURN(ctx->InputDType("num_unique_matrix", 0) == DataType::kUInt32);
  CHECK_OR_RETURN(ctx->InputDType("cur_rank_inverse_indices", 0) == DataType::kUInt32);
  CHECK_OR_RETURN(ctx->InputDType("inverse_unique_partion_indices", 0) == DataType::kUInt32);
  *ctx->OutputDType("embeddings", 0) = ctx->InputDType("cur_rank_embeddings", 0);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> EmbeddingGradientShuffleOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  const Shape& embedding_diff_shape = ctx->InputShape("embedding_diff", 0);
  const Shape& num_unique_matrix_shape = ctx->InputShape("num_unique_matrix", 0);
  const Shape& cur_rank_inverse_indices_shape = ctx->InputShape("cur_rank_inverse_indices", 0);
  const Shape& inverse_unique_partion_indices_shape =
      ctx->InputShape("inverse_unique_partion_indices", 0);
  const int64_t num_ids = inverse_unique_partion_indices_shape.elem_cnt();
  const int64_t parallel_num = ctx->parallel_num();
  const int64_t embedding_size = embedding_diff_shape.elem_cnt() / num_ids;
  CHECK_EQ_OR_RETURN(num_unique_matrix_shape.elem_cnt(), parallel_num * parallel_num);
  CHECK_EQ_OR_RETURN(cur_rank_inverse_indices_shape.elem_cnt(), parallel_num * num_ids);
  DimVector out_dim_vec = cur_rank_inverse_indices_shape.dim_vec();
  out_dim_vec.push_back(embedding_size);
  *ctx->OutputShape("cur_rank_unique_embedding_diff", 0) = Shape(out_dim_vec);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> EmbeddingGradientShuffleOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> EmbeddingGradientShuffleOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Split(ctx->inputs(), 0)
      .Broadcast(user_op::OpArg("num_unique_matrix", 0))
      .Split(ctx->outputs(), 0)
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> EmbeddingGradientShuffleOp::InferDataType(user_op::InferContext* ctx) {
  CHECK_OR_RETURN(ctx->InputDType("num_unique_matrix", 0) == DataType::kUInt32);
  CHECK_OR_RETURN(ctx->InputDType("cur_rank_inverse_indices", 0) == DataType::kUInt32);
  CHECK_OR_RETURN(ctx->InputDType("inverse_unique_partion_indices", 0) == DataType::kUInt32);
  *ctx->OutputDType("cur_rank_unique_embedding_diff", 0) = ctx->InputDType("embedding_diff", 0);
  return Maybe<void>::Ok();
}

}  // namespace oneflow
