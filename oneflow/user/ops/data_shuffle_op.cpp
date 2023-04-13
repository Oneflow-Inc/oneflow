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
#include "oneflow/core/embedding/embedding_manager.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

/* static */ Maybe<void> UniqueKeyValuePairOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& keys_shape = ctx->InputShape("keys", 0);
  const int32_t num_tables = ctx->Attr<int32_t>("num_tables");
  CHECK_GE_OR_RETURN(num_tables, 1) << "num_tables must greater than 1, but get " << num_tables;
  if (ctx->has_input("values", 0)) {
    const Shape& values_shape = ctx->InputShape("values", 0);
    CHECK_EQ_OR_RETURN(keys_shape, values_shape) << "keys_shape must equal to values_shape";
  } else {
    if (num_tables > 1) {
      CHECK_EQ_OR_RETURN(keys_shape.NumAxes(), 2);
      CHECK_EQ_OR_RETURN(keys_shape.At(1), num_tables) << "keys cols must equal to num_tables";
    }
  }
  ctx->SetOutputShape("num_unique", 0, Shape({1}));
  ctx->SetOutputShape("unique_keys", 0, Shape({keys_shape.elem_cnt()}));
  ctx->SetOutputShape("unique_values", 0, Shape({keys_shape.elem_cnt()}));
  ctx->SetOutputShape("inverse_indices", 0, keys_shape);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> UniqueKeyValuePairOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> UniqueKeyValuePairOp::GetSbp(user_op::SbpContext* ctx) {
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> UniqueKeyValuePairOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("num_unique", 0, DataType::kInt32);
  ctx->SetOutputDType("unique_keys", 0, ctx->InputDType("keys", 0));
  ctx->SetOutputDType("inverse_indices", 0, DataType::kInt32);
  if (ctx->has_input("values", 0)) {
    ctx->SetOutputDType("unique_values", 0, ctx->InputDType("values", 0));
  } else {
    ctx->SetOutputDType("unique_values", 0, DataType::kUInt8);
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> IdShuffleOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& ids_shape = ctx->InputShape("ids", 0);
  const int32_t num_tables = ctx->Attr<int32_t>("num_tables");
  CHECK_GE_OR_RETURN(num_tables, 1) << "num_tables must greater than 1, but get " << num_tables;
  if (ctx->has_input("table_ids", 0)) {
    const Shape& table_ids_shape = ctx->InputShape("table_ids", 0);
    CHECK_EQ_OR_RETURN(ids_shape, table_ids_shape) << "ids_shape must equal to table_ids_shape";
  } else {
    if (num_tables > 1) {
      CHECK_EQ_OR_RETURN(ids_shape.NumAxes(), 2);
      CHECK_EQ_OR_RETURN(ids_shape.At(1), num_tables) << "ids cols must equal to num_tables";
    }
  }
  const int64_t num_ids = ids_shape.elem_cnt();
  const int64_t parallel_num = ctx->parallel_num();
  ctx->SetOutputShape("num_unique_matrix", 0, Shape({parallel_num * parallel_num}));
  ctx->SetOutputShape("inverse_unique_partition_indices", 0, ids_shape);
  ctx->SetOutputShape("cur_rank_num_unique", 0, Shape({1}));
  ctx->SetOutputShape("cur_rank_unique_ids", 0, Shape({num_ids * parallel_num}));
  ctx->SetOutputShape("cur_rank_inverse_indices", 0, Shape({num_ids * parallel_num}));
  ctx->SetOutputShape("cur_rank_unique_table_ids", 0, Shape({num_ids * parallel_num}));
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
  ctx->SetOutputDType("num_unique_matrix", 0, DataType::kUInt32);
  ctx->SetOutputDType("inverse_unique_partition_indices", 0, DataType::kUInt32);
  ctx->SetOutputDType("cur_rank_num_unique", 0, DataType::kUInt32);
  ctx->SetOutputDType("cur_rank_unique_ids", 0, ctx->InputDType("ids", 0));
  ctx->SetOutputDType("cur_rank_inverse_indices", 0, DataType::kUInt32);
  if (ctx->has_input("table_ids", 0)) {
    ctx->SetOutputDType("cur_rank_unique_table_ids", 0, ctx->InputDType("table_ids", 0));
  } else {
    ctx->SetOutputDType("cur_rank_unique_table_ids", 0, DataType::kUInt8);
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> EmbeddingShuffleOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& cur_rank_embeddings_shape = ctx->InputShape("cur_rank_embeddings", 0);
  const Shape& num_unique_matrix_shape = ctx->InputShape("num_unique_matrix", 0);
  const Shape& cur_rank_inverse_indices_shape = ctx->InputShape("cur_rank_inverse_indices", 0);
  const Shape& inverse_unique_partition_indices_shape =
      ctx->InputShape("inverse_unique_partition_indices", 0);
  const int64_t embedding_size = ctx->Attr<int64_t>("embedding_size");
  const int64_t num_ids = inverse_unique_partition_indices_shape.elem_cnt();
  const int64_t parallel_num = ctx->parallel_num();
  if (embedding::UseDynamicMemoryAllocation()) {
    CHECK_EQ_OR_RETURN(cur_rank_embeddings_shape.elem_cnt(), 1)
        << "if use dynamic memory allocation, cur_rank_embeddings elem_cnt should be 1.";
  } else {
    CHECK_EQ_OR_RETURN(cur_rank_embeddings_shape.NumAxes(), 2)
        << "cur_rank_embeddings num_axes should be 2.";
    CHECK_EQ_OR_RETURN(cur_rank_embeddings_shape.At(0), parallel_num * num_ids)
        << " got " << cur_rank_embeddings_shape.At(0) << " and " << parallel_num * num_ids;
    CHECK_EQ_OR_RETURN(embedding_size, cur_rank_embeddings_shape.At(1))
        << " got " << embedding_size << " and " << cur_rank_embeddings_shape.At(1);
  }
  CHECK_EQ_OR_RETURN(num_unique_matrix_shape.elem_cnt(), parallel_num * parallel_num);
  CHECK_EQ_OR_RETURN(cur_rank_inverse_indices_shape.elem_cnt(), parallel_num * num_ids);
  DimVector out_dim_vec = inverse_unique_partition_indices_shape.dim_vec();
  out_dim_vec.push_back(embedding_size);
  ctx->SetOutputShape("embeddings", 0, Shape(out_dim_vec));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> EmbeddingShuffleOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> EmbeddingShuffleOp::GetSbp(user_op::SbpContext* ctx) {
  auto builder = ctx->NewBuilder()
                     .Split(ctx->inputs(), 0)
                     .Broadcast(user_op::OpArg("num_unique_matrix", 0))
                     .Split(ctx->outputs(), 0);
  if (embedding::UseDynamicMemoryAllocation()) {
    builder.Broadcast(user_op::OpArg("cur_rank_embeddings", 0)).Build();
  } else {
    builder.Split(user_op::OpArg("cur_rank_embeddings", 0), 0).Build();
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> EmbeddingShuffleOp::InferDataType(user_op::InferContext* ctx) {
  CHECK_OR_RETURN(ctx->InputDType("num_unique_matrix", 0) == DataType::kUInt32);
  CHECK_OR_RETURN(ctx->InputDType("cur_rank_inverse_indices", 0) == DataType::kUInt32);
  CHECK_OR_RETURN(ctx->InputDType("inverse_unique_partition_indices", 0) == DataType::kUInt32);
  ctx->SetOutputDType("embeddings", 0, ctx->InputDType("cur_rank_embeddings", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> EmbeddingGradientShuffleOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  const Shape& embedding_grad_shape = ctx->InputShape("embedding_grad", 0);
  const Shape& num_unique_matrix_shape = ctx->InputShape("num_unique_matrix", 0);
  const Shape& cur_rank_inverse_indices_shape = ctx->InputShape("cur_rank_inverse_indices", 0);
  const Shape& inverse_unique_partition_indices_shape =
      ctx->InputShape("inverse_unique_partition_indices", 0);
  const int64_t num_ids = inverse_unique_partition_indices_shape.elem_cnt();
  const int64_t parallel_num = ctx->parallel_num();
  CHECK_EQ_OR_RETURN(embedding_grad_shape.elem_cnt() % num_ids, 0);
  const int64_t embedding_size = embedding_grad_shape.elem_cnt() / num_ids;
  CHECK_EQ_OR_RETURN(num_unique_matrix_shape.elem_cnt(), parallel_num * parallel_num);
  CHECK_EQ_OR_RETURN(cur_rank_inverse_indices_shape.elem_cnt(), parallel_num * num_ids);
  DimVector out_dim_vec = cur_rank_inverse_indices_shape.dim_vec();
  out_dim_vec.push_back(embedding_size);
  ctx->SetOutputShape("cur_rank_unique_embedding_grad", 0, Shape(out_dim_vec));
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
  CHECK_OR_RETURN(ctx->InputDType("inverse_unique_partition_indices", 0) == DataType::kUInt32);
  ctx->SetOutputDType("cur_rank_unique_embedding_grad", 0, ctx->InputDType("embedding_grad", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> OneEmbeddingGatherOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& in_shape = ctx->InputShape("in", 0);
  const Shape& indices_shape = ctx->InputShape("indices", 0);
  const int64_t embedding_size = ctx->Attr<int64_t>("embedding_size");
  const int64_t num_ids = indices_shape.elem_cnt();
  const int64_t parallel_num = ctx->parallel_num();
  if (embedding::UseDynamicMemoryAllocation()) {
    CHECK_EQ_OR_RETURN(in_shape.elem_cnt(), 1)
        << "if use dynamic memory allocation, in elem_cnt should be 1.";
  } else {
    CHECK_EQ_OR_RETURN(in_shape.NumAxes(), 2) << "in num_axes should be 2.";
    CHECK_EQ_OR_RETURN(in_shape.At(0), parallel_num * num_ids)
        << " got " << in_shape.At(0) << " and " << parallel_num * num_ids;
    CHECK_EQ_OR_RETURN(embedding_size, in_shape.At(1))
        << " got " << embedding_size << " and " << in_shape.At(1);
  }
  DimVector out_dim_vec = indices_shape.dim_vec();
  out_dim_vec.push_back(embedding_size);
  ctx->SetOutputShape("out", 0, Shape(out_dim_vec));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> OneEmbeddingGatherOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> OneEmbeddingGatherOp::GetSbp(user_op::SbpContext* ctx) {
  // Only used in parallel_num = 1.
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> OneEmbeddingGatherOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, ctx->InputDType("in", 0));
  return Maybe<void>::Ok();
}

REGISTER_USER_OP_SAME_OUTPUT_BLOB_REGST_NUM_WITH_FUNC("id_shuffle", []() {
  if (!ParseBooleanFromEnv("ONEFLOW_ONE_EMBEDDING_DISABLE_PIPELINED_EXECUTION", false)) {
    return 2;
  } else {
    return 1;
  }
});

}  // namespace oneflow
