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

/* static */ Maybe<void> EmbeddingPrefetchOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& num_unique_ids_shape = ctx->InputShape("num_unique_ids", 0);
  const Shape& unique_ids_shape = ctx->InputShape("unique_ids", 0);
  const Shape& column_ids_shape = ctx->InputShape("column_ids", 0);
  CHECK_EQ_OR_RETURN(unique_ids_shape, column_ids_shape);
  CHECK_EQ_OR_RETURN(num_unique_ids_shape.elem_cnt(), 1);
  *ctx->OutputShape("context", 0) = num_unique_ids_shape;
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
  const Shape& num_unique_ids_shape = ctx->InputShape("num_unique_ids", 0);
  const Shape& unique_ids_shape = ctx->InputShape("unique_ids", 0);
  const Shape& column_ids_shape = ctx->InputShape("column_ids", 0);
  CHECK_EQ_OR_RETURN(unique_ids_shape, column_ids_shape);
  CHECK_EQ_OR_RETURN(num_unique_ids_shape.elem_cnt(), 1);
  const int64_t embedding_size = ctx->Attr<int64_t>("embedding_size");
  const int64_t line_size = ctx->Attr<int64_t>("line_size");
  CHECK_NE(embedding_size, 0);
  CHECK_NE(line_size, 0);
  CHECK_GE(line_size, embedding_size);
  CHECK_EQ(line_size % embedding_size, 0);
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
  auto builder = ctx->NewBuilder()
                     .Broadcast(user_op::OpArg("num_unique_ids", 0))
                     .Split(user_op::OpArg("unique_ids", 0), 0)
                     .Split(user_op::OpArg("column_ids", 0), 0)
                     .Split(ctx->outputs(), 0);
  if (ctx->user_op_conf().has_input("context", 0)) {
    builder.Broadcast(user_op::OpArg("context", 0));
  }
  builder.Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> EmbeddingLookupOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("unique_values", 0) = ctx->Attr<DataType>("dtype");
  if (ctx->has_output("embeddings", 0)) {
    *ctx->OutputDType("embeddings", 0) = ctx->Attr<DataType>("embeddings_dtype");
  }
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
