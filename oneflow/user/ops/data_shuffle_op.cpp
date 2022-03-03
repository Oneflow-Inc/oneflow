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

/* static */ Maybe<void> UniqueKeyValuePairOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& keys_shape = ctx->InputShape("keys", 0);
  const Shape& values_shape = ctx->InputShape("values", 0);
  CHECK_EQ_OR_RETURN(keys_shape.NumAxes(), values_shape.NumAxes());
  for (int i = 0; i < keys_shape.NumAxes(); ++i) {
    CHECK_EQ_OR_RETURN(keys_shape.At(i), values_shape.At(i));
  }
  *ctx->OutputShape("num_unique", 0) = Shape({1});
  *ctx->OutputShape("unique_keys", 0) = Shape({keys_shape.elem_cnt()});
  *ctx->OutputShape("unique_values", 0) = Shape({values_shape.elem_cnt()});
  *ctx->OutputShape("reverse_index", 0) = keys_shape;
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> UniqueKeyValuePairOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> UniqueKeyValuePairOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder().Split(ctx->inputs(), 0).Split(ctx->outputs(), 0).Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> UniqueKeyValuePairOp::InferDataType(user_op::InferContext* ctx) {
  const DataType key_dtype = ctx->InputDType("keys", 0);
  const DataType values_dtype = ctx->InputDType("values", 0);
  CHECK_EQ_OR_RETURN(values_dtype, DataType::kInt32);
  CHECK_OR_RETURN((key_dtype == DataType::kInt64) || (key_dtype == DataType::kInt32));
  *ctx->OutputDType("num_unique", 0) = DataType::kInt32;
  *ctx->OutputDType("unique_keys", 0) = key_dtype;
  *ctx->OutputDType("unique_values", 0) = values_dtype;
  *ctx->OutputDType("reverse_index", 0) = DataType::kInt32;
  return Maybe<void>::Ok();
}

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
  *ctx->OutputDType("num_unique_matrix", 0) = DataType::kInt32;
  *ctx->OutputDType("inverse_unique_partion_indices", 0) = DataType::kInt32;
  *ctx->OutputDType("cur_rank_num_unique", 0) = DataType::kInt32;
  *ctx->OutputDType("cur_rank_unique_ids", 0) = ctx->InputDType("ids", 0);
  *ctx->OutputDType("cur_rank_inverse_indices", 0) = DataType::kInt32;
  if (ctx->has_input("column_ids", 0)) {
    *ctx->OutputDType("cur_rank_unique_column_ids", 0) = ctx->InputDType("column_ids", 0);
  } else {
    *ctx->OutputDType("cur_rank_unique_column_ids", 0) = DataType::kInt32;
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
  CHECK_OR_RETURN(ctx->InputDType("num_unique_matrix", 0) == DataType::kInt32);
  CHECK_OR_RETURN(ctx->InputDType("cur_rank_inverse_indices", 0) == DataType::kInt32);
  CHECK_OR_RETURN(ctx->InputDType("inverse_unique_partion_indices", 0) == DataType::kInt32);
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
  CHECK_OR_RETURN(ctx->InputDType("num_unique_matrix", 0) == DataType::kInt32);
  CHECK_OR_RETURN(ctx->InputDType("cur_rank_inverse_indices", 0) == DataType::kInt32);
  CHECK_OR_RETURN(ctx->InputDType("inverse_unique_partion_indices", 0) == DataType::kInt32);
  *ctx->OutputDType("cur_rank_unique_embedding_diff", 0) = ctx->InputDType("embedding_diff", 0);
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
      .Broadcast(user_op::OpArg("num_unique_ids", 0))
      .Split(user_op::OpArg("unique_ids", 0), 0)
      .Split(user_op::OpArg("column_ids", 0), 0)
      .Broadcast(user_op::OpArg("context", 0))
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> EmbeddingPrefetchOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("context", 0) = ctx->InputDType("num_unique_ids", 0);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> EmbeddingLookupOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& unique_ids_shape = ctx->InputShape("unique_ids", 0);
  const int64_t embedding_size = ctx->Attr<int64_t>("embedding_size");
  const int64_t line_size = ctx->Attr<int64_t>("line_size");
  CHECK_EQ_OR_RETURN(embedding_size, ParseIntegerFromEnv("EMBEDDING_SIZE", 128));
  if (ctx->has_output("embeddings", 0)) {
    DimVector embeddings_dim_vec = unique_ids_shape.dim_vec();
    embeddings_dim_vec.push_back(embedding_size);
    *ctx->OutputShape("embeddings", 0) = Shape(embeddings_dim_vec);
  }
  DimVector unique_values_dim_vec = unique_ids_shape.dim_vec();
  unique_values_dim_vec.push_back(line_size);
  *ctx->OutputShape("unique_values", 0) = Shape(unique_values_dim_vec);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> EmbeddingLookupOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> EmbeddingLookupOp::GetSbp(user_op::SbpContext* ctx) {
  // ctx->NewBuilder().Split(ctx->inputs(), 0).Split(ctx->outputs(),
  // 0).Broadcast(user_op::OpArg("num_unique_ids", 0)).Build();
  ctx->NewBuilder()
      .Broadcast(user_op::OpArg("num_unique_ids", 0))
      .Split(user_op::OpArg("unique_ids", 0), 0)
      .Split(ctx->outputs(), 0)
      .Broadcast(user_op::OpArg("context", 0))
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> EmbeddingLookupOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("unique_values", 0) = ctx->Attr<DataType>("dtype");
  if (ctx->has_output("embeddings", 0)) {
    *ctx->OutputDType("embeddings", 0) = ctx->Attr<DataType>("embeddings_dtype");
  }
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
      .Broadcast(ctx->inputs())
      .Broadcast(user_op::OpArg("num_unique_ids", 0))
      .Split(user_op::OpArg("unique_embeddings", 0), 0)
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
      .Broadcast(ctx->inputs())
      .Broadcast(user_op::OpArg("num_unique_ids", 0))
      .Split(user_op::OpArg("unique_embeddings", 0), 0)
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
      .Broadcast(user_op::OpArg("num_unique_ids", 0))
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
      .Broadcast(user_op::OpArg("num_unique_ids", 0))
      .Split(user_op::OpArg("unique_ids", 0), 0)
      .Split(user_op::OpArg("unique_embeddings", 0), 0)
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> EmbeddingPutOp::InferDataType(user_op::InferContext* ctx) {
  return Maybe<void>::Ok();
}

}  // namespace oneflow
