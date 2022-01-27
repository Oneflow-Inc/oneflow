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

/* static */ Maybe<void> EmbeddingLookupPlaceholderOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  DimVector out_dim_vec = ctx->InputShape("ids", 0).dim_vec();
  embedding::EmbeddingOptions options(ctx->Attr<std::string>("embedding_options"));
  const int64_t embedding_size = options.EmbeddingSize();
  CHECK_EQ_OR_RETURN(embedding_size, ParseIntegerFromEnv("EMBEDDING_SIZE", 128));
  out_dim_vec.push_back(embedding_size);
  *ctx->OutputShape("embeddings", 0) = Shape(out_dim_vec);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> EmbeddingLookupPlaceholderOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> EmbeddingLookupPlaceholderOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Broadcast(user_op::OpArg("shadow", 0))
      .Split(user_op::OpArg("ids", 0), 0)
      .Split(user_op::OpArg("column_ids", 0), 0)
      .Split(user_op::OpArg("embeddings", 0), 0)
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> EmbeddingLookupPlaceholderOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper& conf) {
  user_op::InputArgModifier* shadow = GetInputArgModifierFn("shadow", 0);
  CHECK_OR_RETURN(shadow != nullptr);
  shadow->set_requires_grad(false);
  user_op::InputArgModifier* ids = GetInputArgModifierFn("ids", 0);
  CHECK_OR_RETURN(ids != nullptr);
  ids->set_requires_grad(false);
  user_op::InputArgModifier* column_ids = GetInputArgModifierFn("column_ids", 0);
  CHECK_OR_RETURN(column_ids != nullptr);
  column_ids->set_requires_grad(false);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> EmbeddingLookupPlaceholderOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("embeddings", 0) = ctx->Attr<DataType>("dtype");
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
      .Split(user_op::OpArg("embedding_diff", 0), 0)
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> EmbeddingUpdatePlaceholderOp::InferDataType(user_op::InferContext* ctx) {
  return Maybe<void>::Ok();
}

REGISTER_USER_OP_GRAD("embedding_lookup_placeholder")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                               user_op::AddOpFn AddOp) -> Maybe<void> {
      user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_update");
      user_op::UserOpConfWrapper grad_op =
          builder.Op("embedding_update_placeholder")
              .Input("ids", op.input("ids", 0))
              .Input("embedding_diff", op.GetGradTensorWithOpOutput("embeddings", 0))
              .Attr<std::string>("embedding_options", op.attr<std::string>("embedding_options"))
              .Build();
      AddOp(grad_op);
      if (op.NeedGenGradTensor4OpInput("shadow", 0)) {
        user_op::UserOpConfWrapperBuilder var_grad_builder(op.op_name() + "_var");
        user_op::UserOpConfWrapper var_grad_op = var_grad_builder.Op("zero_like")
                                                     .Input("like", op.input("shadow", 0))
                                                     .Output("out")
                                                     .Build();
        AddOp(var_grad_op);
        op.BindGradTensorWithOpInput(var_grad_op.output("out", 0), "shadow", 0);
      }
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
