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

namespace oneflow {

REGISTER_USER_OP("id_shuffle")
    .Input("ids")
    .Output("num_unique_ids")
    .Output("ids_reverse_idx")
    .Output("cur_rank_num_unique_ids")
    .Output("cur_rank_unique_ids")
    .Output("cur_rank_reverse_idx")
    .Output("num_unique_ids_matrix")
    .Output("partition_index")
    .SetOutputBufferNum(1)
    .Attr<std::string>("partitioning")
    .SetLogicalTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape& ids_shape = ctx->InputShape("ids", 0);
      const ParallelDesc& parallel_desc = ctx->parallel_desc();
      const int64_t parallel_num = parallel_desc.parallel_num();
      *ctx->OutputShape("num_unique_ids", 0) = Shape({parallel_num});
      *ctx->OutputShape("ids_reverse_idx", 0) = ids_shape;
      *ctx->OutputShape("cur_rank_num_unique_ids", 0) = Shape({parallel_num});
      *ctx->OutputShape("cur_rank_unique_ids", 0) = Shape({ids_shape.elem_cnt() * parallel_num});
      *ctx->OutputShape("cur_rank_reverse_idx", 0) = Shape({ids_shape.elem_cnt() * parallel_num});
      *ctx->OutputShape("num_unique_ids_matrix", 0) = Shape({parallel_num * parallel_num});
      *ctx->OutputShape("partition_index", 0) = Shape({ids_shape.elem_cnt() * parallel_num});

      *ctx->OutputIsDynamic("num_unique_ids", 0) = false;
      *ctx->OutputIsDynamic("cur_rank_num_unique_ids", 0) = false;
      *ctx->OutputIsDynamic("cur_rank_unique_ids", 0) = true;
      *ctx->OutputIsDynamic("cur_rank_reverse_idx", 0) = true;
      *ctx->OutputIsDynamic("ids_reverse_idx", 0) = ctx->InputIsDynamic("ids", 0);
      *ctx->OutputIsDynamic("num_unique_ids_matrix", 0) = false;
      return Maybe<void>::Ok();
    })
    .SetPhysicalTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape& ids_shape = ctx->InputShape("ids", 0);
      const ParallelDesc& parallel_desc = ctx->parallel_desc();
      const int64_t parallel_num = parallel_desc.parallel_num();
      *ctx->OutputShape("num_unique_ids", 0) = Shape({1});
      *ctx->OutputShape("ids_reverse_idx", 0) = ids_shape;
      *ctx->OutputShape("cur_rank_num_unique_ids", 0) = Shape({1});
      *ctx->OutputShape("cur_rank_unique_ids", 0) = Shape({ids_shape.elem_cnt() * parallel_num});
      *ctx->OutputShape("cur_rank_reverse_idx", 0) = Shape({ids_shape.elem_cnt() * parallel_num});
      *ctx->OutputShape("num_unique_ids_matrix", 0) = Shape({parallel_num * parallel_num});
      *ctx->OutputShape("partition_index", 0) = Shape({ids_shape.elem_cnt() * parallel_num});

      *ctx->OutputIsDynamic("num_unique_ids", 0) = false;
      *ctx->OutputIsDynamic("cur_rank_num_unique_ids", 0) = false;
      *ctx->OutputIsDynamic("cur_rank_unique_ids", 0) = true;
      *ctx->OutputIsDynamic("cur_rank_reverse_idx", 0) = true;
      *ctx->OutputIsDynamic("ids_reverse_idx", 0) = ctx->InputIsDynamic("ids", 0);
      *ctx->OutputIsDynamic("num_unique_ids_matrix", 0) = false;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder()
          .Split(user_op::OpArg("ids", 0), 0)
          .Split(user_op::OpArg("num_unique_ids", 0), 0)
          .Split(user_op::OpArg("ids_reverse_idx", 0), 0)
          .Split(user_op::OpArg("cur_rank_num_unique_ids", 0), 0)
          .Split(user_op::OpArg("cur_rank_unique_ids", 0), 0)
          .Split(user_op::OpArg("cur_rank_reverse_idx", 0), 0)
          .Broadcast(user_op::OpArg("num_unique_ids_matrix", 0))
          .Split(user_op::OpArg("partition_index", 0), 0)
          .Build();
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputDType("num_unique_ids", 0) = DataType::kInt32;
      *ctx->OutputDType("ids_reverse_idx", 0) = DataType::kInt32;
      *ctx->OutputDType("cur_rank_num_unique_ids", 0) = DataType::kInt32;
      *ctx->OutputDType("cur_rank_unique_ids", 0) = ctx->InputDType("ids", 0);
      *ctx->OutputDType("cur_rank_reverse_idx", 0) = DataType::kInt32;
      *ctx->OutputDType("num_unique_ids_matrix", 0) = DataType::kInt32;
      *ctx->OutputDType("partition_index", 0) = DataType::kInt32;
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("embedding_prefetch")
    .Input("num_unique_ids")
    .Input("unique_ids")
    .Output("context")
    .SetOutputBufferNum(1)
    .Attr<std::string>("name")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputShape("context", 0) = ctx->InputShape("unique_ids", 0);
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputDType("context", 0) = DataType::kUInt64;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder()
          .Split(user_op::OpArg("num_unique_ids", 0), 0)
          .Split(user_op::OpArg("unique_ids", 0), 0)
          .Split(user_op::OpArg("context", 0), 0)
          .Build();
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("embedding_lookup")
    .Input("num_unique_ids")
    .Input("unique_ids")
    .Input("context")
    .Output("embeddings")
    .Output("out_context")
    .SetOutputBufferNum(1)
    .Attr<int64_t>("embedding_size")
    .Attr<DataType>("dtype")
    .Attr<std::string>("name")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape& ids_shape = ctx->InputShape("unique_ids", 0);
      DimVector out_dim_vec = ids_shape.dim_vec();
      const int64_t embedding_size = ctx->Attr<int64_t>("embedding_size");
      out_dim_vec.push_back(embedding_size);
      *ctx->OutputShape("embeddings", 0) = Shape(out_dim_vec);
      *ctx->OutputShape("out_context", 0) = ctx->InputShape("context", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder()
          .Split(user_op::OpArg("num_unique_ids", 0), 0)
          .Split(user_op::OpArg("context", 0), 0)
          .Split(user_op::OpArg("unique_ids", 0), 0)
          .Split(user_op::OpArg("embeddings", 0), 0)
          .Split(user_op::OpArg("out_context", 0), 0)
          .Build();
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputDType("embeddings", 0) = ctx->Attr<DataType>("dtype");
      *ctx->OutputDType("out_context", 0) = ctx->InputDType("context", 0);
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("sgd_embedding_update")
    .Input("num_unique_ids")
    .Input("unique_ids")
    .Input("context")
    .Input("unique_embeddings")
    .Input("embedding_diff")
    .OptionalInput("learning_rate")
    .Attr<float>("learning_rate_val", 0.0)
    .Attr<std::string>("name")
    .Attr<int64_t>("embedding_size")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> { return Maybe<void>::Ok(); })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder()
          .Split(user_op::OpArg("num_unique_ids", 0), 0)
          .Split(user_op::OpArg("unique_ids", 0), 0)
          .Split(user_op::OpArg("context", 0), 0)
          .Split(user_op::OpArg("unique_embeddings", 0), 0)
          .Broadcast(user_op::OpArg("learning_rate", 0))
          .Split(user_op::OpArg("embedding_diff", 0), 0)
          .Build();
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("embedding_shuffle")
    .Input("cur_rank_embeddings")
    .Input("cur_rank_num_unique_ids")
    .Input("cur_rank_reverse_idx")
    .Input("num_unique_ids")
    .Input("ids_reverse_idx")
    .Input("num_unique_ids_matrix")
    .Input("partition_index")
    .Output("embeddings")
    .Attr<int64_t>("embedding_size")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape& ids_shape = ctx->InputShape("ids_reverse_idx", 0);
      DimVector out_dim_vec = ids_shape.dim_vec();
      const int64_t embedding_size = ctx->Attr<int64_t>("embedding_size");
      out_dim_vec.push_back(embedding_size);
      *ctx->OutputShape("embeddings", 0) = Shape(out_dim_vec);
      *ctx->OutputIsDynamic("embeddings", 0) = false;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
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
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputDType("embeddings", 0) = ctx->InputDType("cur_rank_embeddings", 0);
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("embedding_gradient_shuffle")
    .Input("embedding_diff")
    .Input("cur_rank_num_unique_ids")
    .Input("cur_rank_reverse_idx")
    .Input("num_unique_ids")
    .Input("ids_reverse_idx")
    .Input("num_unique_ids_matrix")
    .Input("partition_index")
    .Output("cur_rank_unique_embedding_diff")
    .Attr<int64_t>("embedding_size")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const ParallelDesc& parallel_desc = ctx->parallel_desc();
      const int64_t parallel_num = parallel_desc.parallel_num();
      const Shape& embedding_diff_shape = ctx->InputShape("embedding_diff", 0);
      DimVector out_dim_vec = embedding_diff_shape.dim_vec();
      out_dim_vec.at(0) *= parallel_num;
      *ctx->OutputShape("cur_rank_unique_embedding_diff", 0) = Shape(out_dim_vec);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
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
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputDType("cur_rank_unique_embedding_diff", 0) = ctx->InputDType("embedding_diff", 0);
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
