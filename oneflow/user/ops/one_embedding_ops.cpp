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

namespace oneflow {

/* static */ Maybe<void> OneEmbeddingFusedLookupOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  const Shape& ids_shape = ctx->InputShape("ids", 0);
  if (ctx->has_input("table_ids", 0)) {
    const Shape& table_ids_shape = ctx->InputShape("table_ids", 0);
    CHECK_EQ_OR_RETURN(ids_shape, table_ids_shape) << "table_ids shape must equal to ids shape";
  }
  DimVector out_dim_vec = ids_shape.dim_vec();
  const int64_t embedding_size = ctx->Attr<int64_t>("embedding_size");
  out_dim_vec.push_back(embedding_size);
  ctx->SetOutputShape("embeddings", 0, Shape(out_dim_vec));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> OneEmbeddingFusedLookupOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> OneEmbeddingFusedLookupOp::GetSbp(user_op::SbpContext* ctx) {
  auto builder = ctx->NewBuilder()
                     .Broadcast(user_op::OpArg("shadow", 0))
                     .Split(user_op::OpArg("ids", 0), 0)
                     .Split(user_op::OpArg("embeddings", 0), 0);
  if (ctx->user_op_conf().has_input("table_ids", 0)) {
    builder.Split(user_op::OpArg("table_ids", 0), 0);
  }
  builder.Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> OneEmbeddingFusedLookupOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper& conf) {
  user_op::InputArgModifier* shadow = GetInputArgModifierFn("shadow", 0);
  CHECK_OR_RETURN(shadow != nullptr) << "shadow is nullptr";
  shadow->set_requires_grad(false);
  user_op::InputArgModifier* ids = GetInputArgModifierFn("ids", 0);
  CHECK_OR_RETURN(ids != nullptr);
  ids->set_requires_grad(false);
  if (conf.has_input("table_ids", 0)) {
    user_op::InputArgModifier* table_ids = GetInputArgModifierFn("table_ids", 0);
    CHECK_OR_RETURN(table_ids != nullptr) << "table_ids is nullptr";
    table_ids->set_requires_grad(false);
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> OneEmbeddingFusedLookupOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("embeddings", 0, ctx->InputDType("shadow", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> OneEmbeddingFusedLookupGradOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> OneEmbeddingFusedLookupGradOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> OneEmbeddingFusedLookupGradOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Split(user_op::OpArg("ids", 0), 0)
      .Split(user_op::OpArg("embedding_grad", 0), 0)
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> OneEmbeddingFusedLookupGradOp::InferDataType(user_op::InferContext* ctx) {
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> EmbeddingPrefetchOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& num_unique_ids_shape = ctx->InputShape("num_unique_ids", 0);
  const Shape& unique_ids_shape = ctx->InputShape("unique_ids", 0);
  const Shape& table_ids_shape = ctx->InputShape("table_ids", 0);
  CHECK_EQ_OR_RETURN(unique_ids_shape, table_ids_shape)
      << "table_ids shape must equal to ids shape";
  CHECK_EQ_OR_RETURN(num_unique_ids_shape.elem_cnt(), 1);
  ctx->SetOutputShape("context", 0, num_unique_ids_shape);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> EmbeddingPrefetchOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> EmbeddingPrefetchOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Broadcast(user_op::OpArg("num_unique_ids", 0))
      .Split(user_op::OpArg("unique_ids", 0), 0)
      .Split(user_op::OpArg("table_ids", 0), 0)
      .Broadcast(user_op::OpArg("context", 0))
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> EmbeddingPrefetchOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("context", 0, ctx->InputDType("num_unique_ids", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> EmbeddingLookupOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& num_unique_ids_shape = ctx->InputShape("num_unique_ids", 0);
  const Shape& unique_ids_shape = ctx->InputShape("unique_ids", 0);
  const Shape& table_ids_shape = ctx->InputShape("table_ids", 0);
  CHECK_EQ_OR_RETURN(unique_ids_shape, table_ids_shape)
      << "table_ids shape must equal to ids shape";
  CHECK_EQ_OR_RETURN(num_unique_ids_shape.elem_cnt(), 1);
  const int64_t embedding_size = ctx->Attr<int64_t>("embedding_size");
  const int64_t line_size = ctx->Attr<int64_t>("line_size");
  CHECK_NE_OR_RETURN(embedding_size, 0);
  CHECK_NE_OR_RETURN(line_size, 0);
  CHECK_GE_OR_RETURN(line_size, embedding_size);
  const bool use_dynamic_memory_allocation = embedding::UseDynamicMemoryAllocation();
  if (ctx->has_output("embeddings", 0)) {
    if (use_dynamic_memory_allocation) {
      ctx->SetOutputShape("embeddings", 0, Shape({1}));
    } else {
      DimVector embeddings_dim_vec = unique_ids_shape.dim_vec();
      embeddings_dim_vec.push_back(embedding_size);
      ctx->SetOutputShape("embeddings", 0, Shape(embeddings_dim_vec));
    }
  }
  if (use_dynamic_memory_allocation) {
    ctx->SetOutputShape("unique_values", 0, Shape({1}));
  } else {
    DimVector unique_values_dim_vec = unique_ids_shape.dim_vec();
    unique_values_dim_vec.push_back(line_size);
    ctx->SetOutputShape("unique_values", 0, Shape(unique_values_dim_vec));
  }

  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> EmbeddingLookupOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> EmbeddingLookupOp::GetSbp(user_op::SbpContext* ctx) {
  auto builder = ctx->NewBuilder()
                     .Broadcast(user_op::OpArg("num_unique_ids", 0))
                     .Split(user_op::OpArg("unique_ids", 0), 0)
                     .Split(user_op::OpArg("table_ids", 0), 0);
  if (ctx->user_op_conf().has_input("context", 0)) {
    builder.Broadcast(user_op::OpArg("context", 0));
  }
  const bool use_dynamic_memory_allocation = embedding::UseDynamicMemoryAllocation();
  if (use_dynamic_memory_allocation) {
    builder.Broadcast(user_op::OpArg("unique_values", 0));
  } else {
    builder.Split(user_op::OpArg("unique_values", 0), 0);
  }
  if (ctx->user_op_conf().has_output("embeddings", 0)) {
    if (use_dynamic_memory_allocation) {
      builder.Broadcast(user_op::OpArg("embeddings", 0));
    } else {
      builder.Split(user_op::OpArg("embeddings", 0), 0);
    }
  }
  builder.Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> EmbeddingLookupOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("unique_values", 0, ctx->Attr<DataType>("dtype"));
  if (ctx->has_output("embeddings", 0)) {
    ctx->SetOutputDType("embeddings", 0, ctx->Attr<DataType>("embeddings_dtype"));
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
  auto builder = ctx->NewBuilder()
                     .Broadcast(user_op::OpArg("num_unique_ids", 0))
                     .Split(user_op::OpArg("unique_ids", 0), 0);
  if (embedding::UseDynamicMemoryAllocation()) {
    builder.Broadcast(user_op::OpArg("unique_embeddings", 0)).Build();
  } else {
    builder.Split(user_op::OpArg("unique_embeddings", 0), 0).Build();
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> EmbeddingPutOp::InferDataType(user_op::InferContext* ctx) {
  return Maybe<void>::Ok();
}

Maybe<void> CheckDataShape(user_op::InferContext* ctx) {
  if (ctx->has_input("learning_rate", 0)) {
    CHECK_EQ_OR_RETURN(ctx->InputShape("learning_rate", 0), Shape({1}));
  }
  if (ctx->has_input("down_scale_by_tensor", 0)) {
    CHECK_EQ_OR_RETURN(ctx->InputShape("down_scale_by_tensor", 0), Shape({1}));
  }
  CHECK_EQ_OR_RETURN(ctx->InputShape("num_unique_ids", 0), Shape({1}));
  const Shape& embedding_grad_shape = ctx->InputShape("embedding_grad", 0);
  CHECK_EQ_OR_RETURN(embedding_grad_shape.NumAxes(), 2);
  const Shape& unique_embeddings_shape = ctx->InputShape("unique_embeddings", 0);
  if (embedding::UseDynamicMemoryAllocation()) {
    CHECK_EQ_OR_RETURN(unique_embeddings_shape.elem_cnt(), 1)
        << "if use dynamic memory allocation, unique_embeddings elem_cnt should be 1.";
  } else {
    CHECK_EQ_OR_RETURN(unique_embeddings_shape.NumAxes(), 2)
        << "unique_embeddings num_axes should be 2.";
    CHECK_EQ_OR_RETURN(unique_embeddings_shape.At(0), embedding_grad_shape.At(0))
        << "got " << unique_embeddings_shape.At(0) << " and " << embedding_grad_shape.At(0);
  }
  return Maybe<void>::Ok();
}

Maybe<void> CheckDataType(user_op::InferContext* ctx) {
  if (ctx->has_input("learning_rate", 0)) {
    const DataType learning_rate_dtype = ctx->InputDType("learning_rate", 0);
    CHECK_EQ_OR_RETURN(learning_rate_dtype, DataType::kFloat)
        << "InferDataType Failed. Expected " << DataType_Name(DataType::kFloat) << ", but got "
        << DataType_Name(learning_rate_dtype);
  }
  if (ctx->has_input("down_scale_by_tensor", 0)) {
    CHECK_EQ_OR_RETURN(ctx->InputDType("down_scale_by_tensor", 0),
                       ctx->InputDType("unique_embeddings", 0))
        << "InferDataType Failed. Expected "
        << DataType_Name(ctx->InputDType("unique_embeddings", 0)) << ", but got "
        << DataType_Name(ctx->InputDType("down_scale_by_tensor", 0));
  }
  return Maybe<void>::Ok();
}

Maybe<void> GetEmbeddingUpdateSbp(user_op::SbpContext* ctx) {
  auto builder = ctx->NewBuilder()
                     .Broadcast(ctx->inputs())
                     .Broadcast(user_op::OpArg("num_unique_ids", 0))
                     .Split(user_op::OpArg("embedding_grad", 0), 0);
  if (embedding::UseDynamicMemoryAllocation()) {
    builder.Broadcast(user_op::OpArg("unique_embeddings", 0))
        .Broadcast(user_op::OpArg("updated_unique_embeddings", 0))
        .Build();
  } else {
    builder.Split(user_op::OpArg("unique_embeddings", 0), 0)
        .Split(user_op::OpArg("updated_unique_embeddings", 0), 0)
        .Build();
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> OneEmbeddingFusedSgdUpdatePutOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> OneEmbeddingFusedSgdUpdatePutOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> OneEmbeddingFusedSgdUpdatePutOp::GetSbp(user_op::SbpContext* ctx) {
  auto builder = ctx->NewBuilder()
                     .Broadcast(user_op::OpArg("learning_rate", 0))
                     .Broadcast(user_op::OpArg("num_unique_ids", 0))
                     .Split(user_op::OpArg("unique_ids", 0), 0)
                     .Split(user_op::OpArg("embedding_grad", 0), 0);
  if (embedding::UseDynamicMemoryAllocation()) {
    builder.Broadcast(user_op::OpArg("unique_embeddings", 0)).Build();
  } else {
    builder.Split(user_op::OpArg("unique_embeddings", 0), 0).Build();
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> OneEmbeddingFusedSgdUpdatePutOp::InferDataType(
    user_op::InferContext* ctx) {
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> OneEmbeddingSgdUpdateOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  JUST(CheckDataShape(ctx));
  const int64_t embedding_size = ctx->Attr<int64_t>("embedding_size");
  const int64_t line_size = ctx->Attr<int64_t>("line_size");
  CHECK_NE_OR_RETURN(embedding_size, 0) << "should set attr embedding_size";
  CHECK_NE_OR_RETURN(line_size, 0) << "should set attr line_size";
  CHECK_EQ_OR_RETURN(line_size, embedding_size)
      << "when use SGD optimizer, line_size should equals to embedding_size, but get line_size: "
      << line_size << " embedding_size: " << embedding_size
      << ", please set size_factor of store_options to 1.";
  const Shape& unique_embeddings_shape = ctx->InputShape("unique_embeddings", 0);
  ctx->SetOutputShape("updated_unique_embeddings", 0, unique_embeddings_shape);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> OneEmbeddingSgdUpdateOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> OneEmbeddingSgdUpdateOp::GetSbp(user_op::SbpContext* ctx) {
  JUST(GetEmbeddingUpdateSbp(ctx));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> OneEmbeddingSgdUpdateOp::InferDataType(user_op::InferContext* ctx) {
  JUST(CheckDataType(ctx));
  ctx->SetOutputDType("updated_unique_embeddings", 0, ctx->InputDType("unique_embeddings", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> OneEmbeddingMomentumUpdateOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  JUST(CheckDataShape(ctx));
  const int64_t embedding_size = ctx->Attr<int64_t>("embedding_size");
  const int64_t line_size = ctx->Attr<int64_t>("line_size");
  CHECK_NE_OR_RETURN(embedding_size, 0) << "should set attr embedding_size";
  CHECK_NE_OR_RETURN(line_size, 0) << "should set attr line_size";
  CHECK_EQ_OR_RETURN(line_size, embedding_size * 2)
      << "when using Momentum optimizer, line_size should equals to embedding_size * 2, but get "
         "line_size: "
      << line_size << " embedding_size: " << embedding_size
      << ", please set size_factor of store_options to 2.";
  const Shape& unique_embeddings_shape = ctx->InputShape("unique_embeddings", 0);
  ctx->SetOutputShape("updated_unique_embeddings", 0, unique_embeddings_shape);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> OneEmbeddingMomentumUpdateOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> OneEmbeddingMomentumUpdateOp::GetSbp(user_op::SbpContext* ctx) {
  JUST(GetEmbeddingUpdateSbp(ctx));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> OneEmbeddingMomentumUpdateOp::InferDataType(user_op::InferContext* ctx) {
  JUST(CheckDataType(ctx));
  ctx->SetOutputDType("updated_unique_embeddings", 0, ctx->InputDType("unique_embeddings", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> OneEmbeddingAdamUpdateOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  JUST(CheckDataShape(ctx));
  const int64_t embedding_size = ctx->Attr<int64_t>("embedding_size");
  const int64_t line_size = ctx->Attr<int64_t>("line_size");
  CHECK_NE_OR_RETURN(embedding_size, 0) << "should set attr embedding_size";
  CHECK_NE_OR_RETURN(line_size, 0) << "should set attr line_size";
  CHECK_EQ_OR_RETURN(line_size, embedding_size * 3)
      << "when using Adam optimizer, line_size should equals to embedding_size * 3, but get "
         "line_size: "
      << line_size << " embedding_size: " << embedding_size
      << ", please set size_factor of store_options to 3.";
  const Shape& unique_embeddings_shape = ctx->InputShape("unique_embeddings", 0);
  ctx->SetOutputShape("updated_unique_embeddings", 0, unique_embeddings_shape);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> OneEmbeddingAdamUpdateOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> OneEmbeddingAdamUpdateOp::GetSbp(user_op::SbpContext* ctx) {
  JUST(GetEmbeddingUpdateSbp(ctx));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> OneEmbeddingAdamUpdateOp::InferDataType(user_op::InferContext* ctx) {
  JUST(CheckDataType(ctx));
  ctx->SetOutputDType("updated_unique_embeddings", 0, ctx->InputDType("unique_embeddings", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> OneEmbeddingSmartDecaySparseAdamUpdateOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  JUST(CheckDataShape(ctx));
  const int64_t embedding_size = ctx->Attr<int64_t>("embedding_size");
  const int64_t line_size = ctx->Attr<int64_t>("line_size");
  CHECK_NE_OR_RETURN(embedding_size, 0) << "should set attr embedding_size";
  CHECK_NE_OR_RETURN(line_size, 0) << "should set attr line_size";
  const int64_t value_dtype_size = GetSizeOfDataType(ctx->InputDType("unique_embeddings", 0));
  const int64_t step_dtype_size = sizeof(int64_t);
  const int64_t model_and_states_bytes = embedding_size * 3 * value_dtype_size;
  const int64_t align_to_step_size_bytes =
      (model_and_states_bytes + step_dtype_size - 1) / step_dtype_size * step_dtype_size;
  const int64_t smart_decay_sparse_adam_line_size =
      (align_to_step_size_bytes + step_dtype_size) / value_dtype_size;
  CHECK_EQ_OR_RETURN(line_size, smart_decay_sparse_adam_line_size)
      << "when using SmartDecayAdam optimizer with embedding_size " << embedding_size
      << ", storage_dim should equals to " << smart_decay_sparse_adam_line_size
      << ", but got "
         "storage_dim: "
      << line_size << ", please set storage_dim of store_options to "
      << smart_decay_sparse_adam_line_size;
  const Shape& unique_embeddings_shape = ctx->InputShape("unique_embeddings", 0);
  ctx->SetOutputShape("updated_unique_embeddings", 0, unique_embeddings_shape);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> OneEmbeddingSmartDecaySparseAdamUpdateOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> OneEmbeddingSmartDecaySparseAdamUpdateOp::GetSbp(
    user_op::SbpContext* ctx) {
  JUST(GetEmbeddingUpdateSbp(ctx));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> OneEmbeddingSmartDecaySparseAdamUpdateOp::InferDataType(
    user_op::InferContext* ctx) {
  JUST(CheckDataType(ctx));
  ctx->SetOutputDType("updated_unique_embeddings", 0, ctx->InputDType("unique_embeddings", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> OneEmbeddingAdagradUpdateOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  JUST(CheckDataShape(ctx));
  const int64_t embedding_size = ctx->Attr<int64_t>("embedding_size");
  const int64_t line_size = ctx->Attr<int64_t>("line_size");
  CHECK_NE_OR_RETURN(embedding_size, 0) << "should set attr embedding_size";
  CHECK_NE_OR_RETURN(line_size, 0) << "should set attr line_size";
  CHECK_EQ_OR_RETURN(line_size, embedding_size * 2)
      << "when using Adagrad optimizer, line_size should equals to embedding_size * 2, but get "
         "line_size: "
      << line_size << " embedding_size: " << embedding_size
      << ", please set size_factor of store_options to 2.";
  const Shape& unique_embeddings_shape = ctx->InputShape("unique_embeddings", 0);
  ctx->SetOutputShape("updated_unique_embeddings", 0, unique_embeddings_shape);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> OneEmbeddingAdagradUpdateOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> OneEmbeddingAdagradUpdateOp::GetSbp(user_op::SbpContext* ctx) {
  JUST(GetEmbeddingUpdateSbp(ctx));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> OneEmbeddingAdagradUpdateOp::InferDataType(user_op::InferContext* ctx) {
  JUST(CheckDataType(ctx));
  ctx->SetOutputDType("updated_unique_embeddings", 0, ctx->InputDType("unique_embeddings", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> OneEmbeddingFtrlUpdateOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  JUST(CheckDataShape(ctx));
  const int64_t embedding_size = ctx->Attr<int64_t>("embedding_size");
  const int64_t line_size = ctx->Attr<int64_t>("line_size");
  CHECK_NE_OR_RETURN(embedding_size, 0) << "should set attr embedding_size";
  CHECK_NE_OR_RETURN(line_size, 0) << "should set attr line_size";
  CHECK_EQ_OR_RETURN(line_size, embedding_size * 3)
      << "when using Ftrl optimizer, line_size should equals to embedding_size * 3, but get "
         "line_size: "
      << line_size << " embedding_size: " << embedding_size
      << ", please set size_factor of store_options to 3.";
  const Shape& unique_embeddings_shape = ctx->InputShape("unique_embeddings", 0);
  ctx->SetOutputShape("updated_unique_embeddings", 0, unique_embeddings_shape);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> OneEmbeddingFtrlUpdateOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> OneEmbeddingFtrlUpdateOp::GetSbp(user_op::SbpContext* ctx) {
  JUST(GetEmbeddingUpdateSbp(ctx));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> OneEmbeddingFtrlUpdateOp::InferDataType(user_op::InferContext* ctx) {
  JUST(CheckDataType(ctx));
  ctx->SetOutputDType("updated_unique_embeddings", 0, ctx->InputDType("unique_embeddings", 0));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> IdShuffleCopyOutOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Split(ctx->inputs(), 0)
      .Split(ctx->outputs(), 0)
      .Broadcast(user_op::OpArg("num_unique_matrix", 0))
      .Broadcast(user_op::OpArg("out_num_unique_matrix", 0))
      .Broadcast(user_op::OpArg("cur_rank_num_unique", 0))
      .Broadcast(user_op::OpArg("out_cur_rank_num_unique", 0))
      .Build();
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> IdShuffleCopyOutOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  ctx->SetOutputShape("out_num_unique_matrix", 0, ctx->InputShape("num_unique_matrix", 0));
  ctx->SetOutputShape("out_inverse_unique_partition_indices", 0,
                      ctx->InputShape("inverse_unique_partition_indices", 0));
  ctx->SetOutputShape("out_cur_rank_num_unique", 0, ctx->InputShape("cur_rank_num_unique", 0));
  ctx->SetOutputShape("out_cur_rank_unique_ids", 0, ctx->InputShape("cur_rank_unique_ids", 0));
  ctx->SetOutputShape("out_cur_rank_unique_table_ids", 0,
                      ctx->InputShape("cur_rank_unique_table_ids", 0));
  ctx->SetOutputShape("out_cur_rank_inverse_indices", 0,
                      ctx->InputShape("cur_rank_inverse_indices", 0));
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> IdShuffleCopyOutOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> IdShuffleCopyOutOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out_num_unique_matrix", 0, ctx->InputDType("num_unique_matrix", 0));
  ctx->SetOutputDType("out_inverse_unique_partition_indices", 0,
                      ctx->InputDType("inverse_unique_partition_indices", 0));
  ctx->SetOutputDType("out_cur_rank_num_unique", 0, ctx->InputDType("cur_rank_num_unique", 0));
  ctx->SetOutputDType("out_cur_rank_unique_ids", 0, ctx->InputDType("cur_rank_unique_ids", 0));
  ctx->SetOutputDType("out_cur_rank_unique_table_ids", 0,
                      ctx->InputDType("cur_rank_unique_table_ids", 0));
  ctx->SetOutputDType("out_cur_rank_inverse_indices", 0,
                      ctx->InputDType("cur_rank_inverse_indices", 0));
  return Maybe<void>::Ok();
}

}  // namespace oneflow
