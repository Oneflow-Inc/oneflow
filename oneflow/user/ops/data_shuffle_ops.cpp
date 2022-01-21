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
#include "oneflow/core/embedding/embedding_options.h"

namespace oneflow {

/* static */ Maybe<void> IdShuffleOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& ids_shape = ctx->InputShape("ids", 0);
  const Shape& column_ids_shape = ctx->InputShape("column_ids", 0);
  CHECK_EQ_OR_RETURN(ids_shape, column_ids_shape);
  const ParallelDesc& parallel_desc = ctx->parallel_desc();
  const int64_t parallel_num = parallel_desc.parallel_num();
  *ctx->OutputShape("num_unique_ids", 0) = Shape({parallel_num});
  *ctx->OutputShape("ids_reverse_idx", 0) = ids_shape;
  *ctx->OutputShape("cur_rank_num_unique_ids", 0) = Shape({parallel_num});
  *ctx->OutputShape("cur_rank_unique_ids", 0) = Shape({ids_shape.elem_cnt() * parallel_num});
  *ctx->OutputShape("cur_rank_column_ids", 0) = Shape({ids_shape.elem_cnt() * parallel_num});
  *ctx->OutputShape("cur_rank_reverse_idx", 0) = Shape({ids_shape.elem_cnt() * parallel_num});
  *ctx->OutputShape("num_unique_ids_matrix", 0) = Shape({parallel_num * parallel_num});
  *ctx->OutputShape("partition_index", 0) = Shape({ids_shape.elem_cnt() * parallel_num});

  *ctx->OutputIsDynamic("num_unique_ids", 0) = false;
  *ctx->OutputIsDynamic("cur_rank_num_unique_ids", 0) = false;
  *ctx->OutputIsDynamic("cur_rank_unique_ids", 0) = false;
  *ctx->OutputIsDynamic("cur_rank_column_ids", 0) = false;
  *ctx->OutputIsDynamic("cur_rank_reverse_idx", 0) = false;
  *ctx->OutputIsDynamic("ids_reverse_idx", 0) = ctx->InputIsDynamic("ids", 0);
  *ctx->OutputIsDynamic("num_unique_ids_matrix", 0) = false;
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> IdShuffleOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& ids_shape = ctx->InputShape("ids", 0);
  const Shape& column_ids_shape = ctx->InputShape("column_ids", 0);
  CHECK_EQ_OR_RETURN(ids_shape, column_ids_shape);
  const ParallelDesc& parallel_desc = ctx->parallel_desc();
  const int64_t parallel_num = parallel_desc.parallel_num();
  *ctx->OutputShape("num_unique_ids", 0) = Shape({1});
  *ctx->OutputShape("ids_reverse_idx", 0) = ids_shape;
  *ctx->OutputShape("cur_rank_num_unique_ids", 0) = Shape({1});
  *ctx->OutputShape("cur_rank_unique_ids", 0) = Shape({ids_shape.elem_cnt() * parallel_num});
  *ctx->OutputShape("cur_rank_column_ids", 0) = Shape({ids_shape.elem_cnt() * parallel_num});
  *ctx->OutputShape("cur_rank_reverse_idx", 0) = Shape({ids_shape.elem_cnt() * parallel_num});
  *ctx->OutputShape("num_unique_ids_matrix", 0) = Shape({parallel_num * parallel_num});
  *ctx->OutputShape("partition_index", 0) = Shape({ids_shape.elem_cnt() * parallel_num});

  // can't set to dynamic when need boxing
  *ctx->OutputIsDynamic("num_unique_ids", 0) = false;
  *ctx->OutputIsDynamic("cur_rank_num_unique_ids", 0) = false;
  *ctx->OutputIsDynamic("cur_rank_unique_ids", 0) = false;
  *ctx->OutputIsDynamic("cur_rank_column_ids", 0) = false;
  *ctx->OutputIsDynamic("cur_rank_reverse_idx", 0) = false;
  *ctx->OutputIsDynamic("ids_reverse_idx", 0) = ctx->InputIsDynamic("ids", 0);
  *ctx->OutputIsDynamic("num_unique_ids_matrix", 0) = false;
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> IdShuffleOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Split(user_op::OpArg("ids", 0), 0)
      .Split(user_op::OpArg("column_ids", 0), 0)
      .Split(user_op::OpArg("num_unique_ids", 0), 0)
      .Split(user_op::OpArg("ids_reverse_idx", 0), 0)
      .Split(user_op::OpArg("cur_rank_num_unique_ids", 0), 0)
      .Split(user_op::OpArg("cur_rank_unique_ids", 0), 0)
      .Split(user_op::OpArg("cur_rank_column_ids", 0), 0)
      .Split(user_op::OpArg("cur_rank_reverse_idx", 0), 0)
      .Broadcast(user_op::OpArg("num_unique_ids_matrix", 0))
      .Split(user_op::OpArg("partition_index", 0), 0)
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> IdShuffleOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("num_unique_ids", 0) = DataType::kInt32;
  *ctx->OutputDType("ids_reverse_idx", 0) = DataType::kInt32;
  *ctx->OutputDType("cur_rank_num_unique_ids", 0) = DataType::kInt32;
  *ctx->OutputDType("cur_rank_unique_ids", 0) = ctx->InputDType("ids", 0);
  *ctx->OutputDType("cur_rank_column_ids", 0) = ctx->InputDType("column_ids", 0);
  *ctx->OutputDType("cur_rank_reverse_idx", 0) = DataType::kInt32;
  *ctx->OutputDType("num_unique_ids_matrix", 0) = DataType::kInt32;
  *ctx->OutputDType("partition_index", 0) = DataType::kInt32;
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> EmbeddingPrefetchOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  *ctx->OutputShape("context", 0) = ctx->InputShape("num_unique_ids", 0);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> EmbeddingPrefetchOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> EmbeddingPrefetchOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Split(user_op::OpArg("num_unique_ids", 0), 0)
      .Split(user_op::OpArg("unique_ids", 0), 0)
      .Split(user_op::OpArg("column_ids", 0), 0)
      .Split(user_op::OpArg("context", 0), 0)
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> EmbeddingPrefetchOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("context", 0) = ctx->InputDType("num_unique_ids", 0);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> EmbeddingLookupOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& ids_shape = ctx->InputShape("unique_ids", 0);
  DimVector out_dim_vec = ids_shape.dim_vec();
  embedding::EmbeddingOptions options(ctx->Attr<std::string>("embedding_options"));
  const int64_t embedding_size = options.EmbeddingSize();
  CHECK_EQ_OR_RETURN(embedding_size, ParseIntegerFromEnv("EMBEDDING_SIZE", 128));
  out_dim_vec.push_back(embedding_size);
  *ctx->OutputShape("embeddings", 0) = Shape(out_dim_vec);
  out_dim_vec.at(out_dim_vec.size() - 1) = options.LineSize();
  *ctx->OutputShape("unique_values", 0) = Shape(out_dim_vec);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> EmbeddingLookupOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> EmbeddingLookupOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Split(user_op::OpArg("num_unique_ids", 0), 0)
      .Split(user_op::OpArg("context", 0), 0)
      .Split(user_op::OpArg("unique_ids", 0), 0)
      .Split(user_op::OpArg("unique_values", 0), 0)
      .Split(user_op::OpArg("embeddings", 0), 0)
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> EmbeddingLookupOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("unique_values", 0) = ctx->Attr<DataType>("dtype");
  *ctx->OutputDType("embeddings", 0) = ctx->Attr<DataType>("dtype");
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> SgdEmbeddingUpdateOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  *ctx->OutputShape("updated_unique_embeddings", 0) = ctx->InputShape("unique_embeddings", 0);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> SgdEmbeddingUpdateOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> SgdEmbeddingUpdateOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Split(user_op::OpArg("num_unique_ids", 0), 0)
      .Split(user_op::OpArg("unique_embeddings", 0), 0)
      .Broadcast(user_op::OpArg("learning_rate", 0))
      .Broadcast(user_op::OpArg("skip_if", 0))
      .Split(user_op::OpArg("embedding_diff", 0), 0)
      .Split(user_op::OpArg("updated_unique_embeddings", 0), 0)
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> SgdEmbeddingUpdateOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("updated_unique_embeddings", 0) = ctx->InputDType("unique_embeddings", 0);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> MomentumEmbeddingUpdateOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  *ctx->OutputShape("updated_unique_embeddings", 0) = ctx->InputShape("unique_embeddings", 0);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> MomentumEmbeddingUpdateOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> MomentumEmbeddingUpdateOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Split(user_op::OpArg("num_unique_ids", 0), 0)
      .Split(user_op::OpArg("unique_embeddings", 0), 0)
      .Broadcast(user_op::OpArg("learning_rate", 0))
      .Broadcast(user_op::OpArg("skip_if", 0))
      .Split(user_op::OpArg("embedding_diff", 0), 0)
      .Split(user_op::OpArg("updated_unique_embeddings", 0), 0)
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> MomentumEmbeddingUpdateOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("updated_unique_embeddings", 0) = ctx->InputDType("unique_embeddings", 0);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> AdamEmbeddingUpdateOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  *ctx->OutputShape("updated_unique_embeddings", 0) = ctx->InputShape("unique_embeddings", 0);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> AdamEmbeddingUpdateOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> AdamEmbeddingUpdateOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Broadcast(ctx->inputs())
      .Split(user_op::OpArg("num_unique_ids", 0), 0)
      .Split(user_op::OpArg("unique_embeddings", 0), 0)
      .Split(user_op::OpArg("embedding_diff", 0), 0)
      .Split(user_op::OpArg("updated_unique_embeddings", 0), 0)
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> AdamEmbeddingUpdateOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("updated_unique_embeddings", 0) = ctx->InputDType("unique_embeddings", 0);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> EmbeddingPutOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> EmbeddingPutOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> EmbeddingPutOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Split(user_op::OpArg("num_unique_ids", 0), 0)
      .Split(user_op::OpArg("unique_ids", 0), 0)
      .Split(user_op::OpArg("unique_embeddings", 0), 0)
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> EmbeddingPutOp::InferDataType(user_op::InferContext* ctx) {
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> EmbeddingShuffleOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& ids_shape = ctx->InputShape("ids_reverse_idx", 0);
  DimVector out_dim_vec = ids_shape.dim_vec();
  embedding::EmbeddingOptions options(ctx->Attr<std::string>("embedding_options"));
  const int64_t embedding_size = options.EmbeddingSize();
  CHECK_EQ_OR_RETURN(embedding_size, ParseIntegerFromEnv("EMBEDDING_SIZE", 128));
  out_dim_vec.push_back(embedding_size);
  *ctx->OutputShape("embeddings", 0) = Shape(out_dim_vec);
  *ctx->OutputIsDynamic("embeddings", 0) = false;
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> EmbeddingShuffleOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> EmbeddingShuffleOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Broadcast(user_op::OpArg("num_unique_ids_matrix", 0))
      .Split(user_op::OpArg("cur_rank_embeddings", 0), 0)
      .Split(user_op::OpArg("cur_rank_num_unique_ids", 0), 0)
      .Split(user_op::OpArg("cur_rank_reverse_idx", 0), 0)
      .Split(user_op::OpArg("num_unique_ids", 0), 0)
      .Split(user_op::OpArg("ids_reverse_idx", 0), 0)
      .Split(user_op::OpArg("embeddings", 0), 0)
      .Split(user_op::OpArg("partition_index", 0), 0)
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> EmbeddingShuffleOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("embeddings", 0) = ctx->InputDType("cur_rank_embeddings", 0);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> EmbeddingGradientShuffleOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  const ParallelDesc& parallel_desc = ctx->parallel_desc();
  const int64_t parallel_num = parallel_desc.parallel_num();
  const Shape& embedding_diff_shape = ctx->InputShape("embedding_diff", 0);
  DimVector out_dim_vec = embedding_diff_shape.dim_vec();
  out_dim_vec.at(0) *= parallel_num;
  *ctx->OutputShape("cur_rank_unique_embedding_diff", 0) = Shape(out_dim_vec);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> EmbeddingGradientShuffleOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> EmbeddingGradientShuffleOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Broadcast(user_op::OpArg("num_unique_ids_matrix", 0))
      .Split(user_op::OpArg("cur_rank_num_unique_ids", 0), 0)
      .Split(user_op::OpArg("cur_rank_reverse_idx", 0), 0)
      .Split(user_op::OpArg("num_unique_ids", 0), 0)
      .Split(user_op::OpArg("ids_reverse_idx", 0), 0)
      .Split(user_op::OpArg("embedding_diff", 0), 0)
      .Split(user_op::OpArg("cur_rank_unique_embedding_diff", 0), 0)
      .Split(user_op::OpArg("partition_index", 0), 0)
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> EmbeddingGradientShuffleOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("cur_rank_unique_embedding_diff", 0) = ctx->InputDType("embedding_diff", 0);
  return Maybe<void>::Ok();
}
}  // namespace oneflow
