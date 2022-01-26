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

/* static */ Maybe<void> FusedInteractionOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& dense_feature_shape = ctx->InputShape("dense_feature", 0);
  const Shape& sparse_feature_shape = ctx->InputShape("sparse_feature", 0);
  CHECK_EQ(dense_feature_shape.NumAxes(), 2);
  CHECK_EQ(sparse_feature_shape.NumAxes(), 3);
  const int64_t batch_size = dense_feature_shape.At(0);
  const int64_t embedding_size = dense_feature_shape.At(1);
  CHECK_EQ_OR_RETURN(batch_size, sparse_feature_shape.At(0));
  CHECK_EQ_OR_RETURN(embedding_size, sparse_feature_shape.At(2));
  const int64_t num_columns = sparse_feature_shape.At(1);
  const int64_t pad_dim = std::ceil((1 + num_columns) / static_cast<float>(8)) * 8;
  const int64_t out_dim =
      std::ceil((num_columns * (num_columns + 1) / 2 + embedding_size) / static_cast<float>(8)) * 8;
  *ctx->OutputShape("out", 0) = Shape({batch_size, out_dim});
  *ctx->OutputShape("concat_out", 0) = Shape({batch_size, pad_dim, embedding_size});
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> FusedInteractionOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> FusedInteractionOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Split(user_op::OpArg("dense_feature", 0), 0)
      .Split(user_op::OpArg("sparse_feature", 0), 0)
      .Split(user_op::OpArg("out", 0), 0)
      .Split(user_op::OpArg("concat_out", 0), 0)
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> FusedInteractionOp::InferDataType(user_op::InferContext* ctx) {
  CHECK_EQ_OR_RETURN(ctx->InputDType("dense_feature", 0), ctx->InputDType("sparse_feature", 0));
  *ctx->OutputDType("out", 0) = ctx->InputDType("dense_feature", 0);
  *ctx->OutputDType("concat_out", 0) = ctx->InputDType("dense_feature", 0);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> FusedInteractionGradOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  const Shape& dy_shape = ctx->InputShape("dy", 0);
  const int64_t batch_size = dy_shape.At(0);
  const int64_t embedding_size = ctx->Attr<int64_t>("embedding_size");
  const int64_t num_columns = ctx->Attr<int64_t>("num_columns");
  *ctx->OutputShape("dense_feature_grad", 0) = Shape({batch_size, embedding_size});
  *ctx->OutputShape("sparse_feature_grad", 0) = Shape({batch_size, num_columns, embedding_size});
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> FusedInteractionGradOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> FusedInteractionGradOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Split(user_op::OpArg("dy", 0), 0)
      .Split(user_op::OpArg("concat_out", 0), 0)
      .Split(user_op::OpArg("dense_feature_grad", 0), 0)
      .Split(user_op::OpArg("sparse_feature_grad", 0), 0)
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> FusedInteractionGradOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("dense_feature_grad", 0) = ctx->InputDType("dy", 0);
  *ctx->OutputDType("sparse_feature_grad", 0) = ctx->InputDType("dy", 0);
  return Maybe<void>::Ok();
}

REGISTER_USER_OP_GRAD("fused_interaction")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                               user_op::AddOpFn AddOp) -> Maybe<void> {
      const Shape& sparse_feature_shape =
          op.TensorDesc4ArgNameAndIndex("sparse_feature", 0).shape();
      CHECK_EQ_OR_RETURN(sparse_feature_shape.NumAxes(), 3);
      const int64_t embedding_size = sparse_feature_shape.At(2);
      const int64_t num_columns = sparse_feature_shape.At(1);
      user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
      auto grad_op = builder.Op("fused_interaction_grad")
                         .Input("dy", op.GetGradTensorWithOpOutput("out", 0))
                         .Input("concat_out", op.output("concat_out", 0))
                         .Output("dense_feature_grad")
                         .Output("sparse_feature_grad")
                         .Attr<int64_t>("embedding_size", embedding_size)
                         .Attr<int64_t>("num_columns", num_columns)
                         .Build();
      AddOp(grad_op);
      if (op.NeedGenGradTensor4OpInput("dense_feature", 0)) {
        op.BindGradTensorWithOpInput(grad_op.output("dense_feature_grad", 0), "dense_feature", 0);
      }
      if (op.NeedGenGradTensor4OpInput("sparse_feature", 0)) {
        op.BindGradTensorWithOpInput(grad_op.output("sparse_feature_grad", 0), "sparse_feature", 0);
      }
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
