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

/* static */ Maybe<void> EmbeddingLookupPlaceholderOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  const Shape& ids_shape = ctx->InputShape("ids", 0);
  if (ctx->has_input("table_ids", 0)) {
    const Shape& table_ids_shape = ctx->InputShape("table_ids", 0);
    CHECK_EQ_OR_RETURN(ids_shape, table_ids_shape) << "table_ids shape must equal to ids shape";
  }
  DimVector out_dim_vec = ids_shape.dim_vec();
  const int64_t embedding_size = ctx->Attr<int64_t>("embedding_size");
  out_dim_vec.push_back(embedding_size);
  *ctx->OutputShape("embeddings", 0) = Shape(out_dim_vec);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> EmbeddingLookupPlaceholderOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> EmbeddingLookupPlaceholderOp::GetSbp(user_op::SbpContext* ctx) {
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

/* static */ Maybe<void> EmbeddingLookupPlaceholderOp::ModifyInputArg(
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

/* static */ Maybe<void> EmbeddingLookupPlaceholderOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("embeddings", 0) = ctx->InputDType("shadow", 0);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> EmbeddingUpdatePlaceholderOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> EmbeddingUpdatePlaceholderOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> EmbeddingUpdatePlaceholderOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Split(user_op::OpArg("ids", 0), 0)
      .Split(user_op::OpArg("embedding_grad", 0), 0)
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> EmbeddingUpdatePlaceholderOp::InferDataType(user_op::InferContext* ctx) {
  return Maybe<void>::Ok();
}

REGISTER_USER_OP_GRAD("embedding_lookup_placeholder")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                               const user_op::AddOpFn& AddOp) -> Maybe<void> {
      user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_update");
      user_op::UserOpConfWrapper grad_op =
          builder.Op("embedding_update_placeholder")
              .Input("ids", op.input("ids", 0))
              .Input("embedding_grad", op.GetGradTensorWithOpOutput("embeddings", 0))
              .Attr<std::string>("key_value_store_options",
                                 op.attr<std::string>("key_value_store_options"))
              .Build();
      AddOp(grad_op);
      return Maybe<void>::Ok();
    });

/* static */ Maybe<void> EmbeddingPrefetchOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& num_unique_ids_shape = ctx->InputShape("num_unique_ids", 0);
  const Shape& unique_ids_shape = ctx->InputShape("unique_ids", 0);
  const Shape& table_ids_shape = ctx->InputShape("table_ids", 0);
  CHECK_EQ_OR_RETURN(unique_ids_shape, table_ids_shape)
      << "table_ids shape must equal to ids shape";
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
      .Split(user_op::OpArg("table_ids", 0), 0)
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
  const Shape& table_ids_shape = ctx->InputShape("table_ids", 0);
  CHECK_EQ_OR_RETURN(unique_ids_shape, table_ids_shape)
      << "table_ids shape must equal to ids shape";
  CHECK_EQ_OR_RETURN(num_unique_ids_shape.elem_cnt(), 1);
  const int64_t embedding_size = ctx->Attr<int64_t>("embedding_size");
  const int64_t line_size = ctx->Attr<int64_t>("line_size");
  CHECK_NE_OR_RETURN(embedding_size, 0);
  CHECK_NE_OR_RETURN(line_size, 0);
  CHECK_GE_OR_RETURN(line_size, embedding_size);
  CHECK_EQ_OR_RETURN(line_size % embedding_size, 0);
  const bool use_dynamic_memory_allocation = embedding::UseDynamicMemoryAllocation();
  if (ctx->has_output("embeddings", 0)) {
    if (use_dynamic_memory_allocation) {
      *ctx->OutputShape("embeddings", 0) = Shape({1});
    } else {
      DimVector embeddings_dim_vec = unique_ids_shape.dim_vec();
      embeddings_dim_vec.push_back(embedding_size);
      *ctx->OutputShape("embeddings", 0) = Shape(embeddings_dim_vec);
    }
  }
  if (use_dynamic_memory_allocation) {
    *ctx->OutputShape("unique_values", 0) = Shape({1});
  } else {
    DimVector unique_values_dim_vec = unique_ids_shape.dim_vec();
    unique_values_dim_vec.push_back(line_size);
    *ctx->OutputShape("unique_values", 0) = Shape(unique_values_dim_vec);
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
  CHECK_EQ_OR_RETURN(ctx->InputShape("learning_rate", 0), Shape({1}));
  if (ctx->has_input("down_scale_by_tensor", 0)) {
    CHECK_EQ_OR_RETURN(ctx->InputShape("learning_rate", 0), Shape({1}));
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
  const DataType learning_rate_dtype = ctx->InputDType("learning_rate", 0);
  CHECK_EQ_OR_RETURN(learning_rate_dtype, DataType::kFloat);
  if (ctx->has_input("down_scale_by_tensor", 0)) {
    CHECK_EQ_OR_RETURN(ctx->InputDType("down_scale_by_tensor", 0),
                       ctx->InputDType("unique_embeddings", 0));
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

/* static */ Maybe<void> SgdEmbeddingUpdateOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  JUST(CheckDataShape(ctx));
  const Shape& unique_embeddings_shape = ctx->InputShape("unique_embeddings", 0);
  *ctx->OutputShape("updated_unique_embeddings", 0) = unique_embeddings_shape;
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> SgdEmbeddingUpdateOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> SgdEmbeddingUpdateOp::GetSbp(user_op::SbpContext* ctx) {
  JUST(GetEmbeddingUpdateSbp(ctx));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> SgdEmbeddingUpdateOp::InferDataType(user_op::InferContext* ctx) {
  JUST(CheckDataType(ctx));
  *ctx->OutputDType("updated_unique_embeddings", 0) = ctx->InputDType("unique_embeddings", 0);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> MomentumEmbeddingUpdateOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  JUST(CheckDataShape(ctx));
  const Shape& unique_embeddings_shape = ctx->InputShape("unique_embeddings", 0);
  *ctx->OutputShape("updated_unique_embeddings", 0) = unique_embeddings_shape;
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> MomentumEmbeddingUpdateOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> MomentumEmbeddingUpdateOp::GetSbp(user_op::SbpContext* ctx) {
  JUST(GetEmbeddingUpdateSbp(ctx));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> MomentumEmbeddingUpdateOp::InferDataType(user_op::InferContext* ctx) {
  JUST(CheckDataType(ctx));
  *ctx->OutputDType("updated_unique_embeddings", 0) = ctx->InputDType("unique_embeddings", 0);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> AdamEmbeddingUpdateOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  JUST(CheckDataShape(ctx));
  const Shape& unique_embeddings_shape = ctx->InputShape("unique_embeddings", 0);
  *ctx->OutputShape("updated_unique_embeddings", 0) = unique_embeddings_shape;
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> AdamEmbeddingUpdateOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> AdamEmbeddingUpdateOp::GetSbp(user_op::SbpContext* ctx) {
  JUST(GetEmbeddingUpdateSbp(ctx));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> AdamEmbeddingUpdateOp::InferDataType(user_op::InferContext* ctx) {
  JUST(CheckDataType(ctx));
  *ctx->OutputDType("updated_unique_embeddings", 0) = ctx->InputDType("unique_embeddings", 0);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> AdagradEmbeddingUpdateOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  JUST(CheckDataShape(ctx));
  const Shape& unique_embeddings_shape = ctx->InputShape("unique_embeddings", 0);
  *ctx->OutputShape("updated_unique_embeddings", 0) = unique_embeddings_shape;
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> AdagradEmbeddingUpdateOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> AdagradEmbeddingUpdateOp::GetSbp(user_op::SbpContext* ctx) {
  JUST(GetEmbeddingUpdateSbp(ctx));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> AdagradEmbeddingUpdateOp::InferDataType(user_op::InferContext* ctx) {
  JUST(CheckDataType(ctx));
  *ctx->OutputDType("updated_unique_embeddings", 0) = ctx->InputDType("unique_embeddings", 0);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> FtrlEmbeddingUpdateOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  JUST(CheckDataShape(ctx));
  const Shape& unique_embeddings_shape = ctx->InputShape("unique_embeddings", 0);
  *ctx->OutputShape("updated_unique_embeddings", 0) = unique_embeddings_shape;
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> FtrlEmbeddingUpdateOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> FtrlEmbeddingUpdateOp::GetSbp(user_op::SbpContext* ctx) {
  JUST(GetEmbeddingUpdateSbp(ctx));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> FtrlEmbeddingUpdateOp::InferDataType(user_op::InferContext* ctx) {
  JUST(CheckDataType(ctx));
  *ctx->OutputDType("updated_unique_embeddings", 0) = ctx->InputDType("unique_embeddings", 0);
  return Maybe<void>::Ok();
}

}  // namespace oneflow
