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

/* static */ Maybe<void> FusedDotFeatureInteractionOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  const int64_t feature_input_size = ctx->input_size("features");
  CHECK_GE_OR_RETURN(feature_input_size, 1);
  const Shape& first_feature_shape = ctx->InputShape("features", 0);
  CHECK_EQ_OR_RETURN(first_feature_shape.NumAxes(), 3);
  const int64_t batch_size = first_feature_shape.At(0);
  const int64_t vector_size = first_feature_shape.At(2);
  int64_t features_concated_dim = first_feature_shape.At(1);
  for (int64_t i = 1; i < feature_input_size; ++i) {
    const Shape& feature_shape = ctx->InputShape("features", i);
    CHECK_EQ_OR_RETURN(feature_shape.NumAxes(), 3);
    CHECK_EQ_OR_RETURN(feature_shape.At(0), batch_size);
    CHECK_EQ_OR_RETURN(feature_shape.At(2), vector_size);
    features_concated_dim += feature_shape.At(1);
  }
  const std::string& pooling = ctx->Attr<std::string>("pooling");
  if (pooling == "sum") {
    ctx->SetOutputShape("out", 0, Shape({batch_size, vector_size}));
    return Maybe<void>::Ok();
  }
  if (ctx->has_input("sparse_feature", 0)) {
    CHECK_OR_RETURN(pooling == "none") << "only none pooling support sparse feature.";
    CHECK_OR_RETURN(ctx->has_input("sparse_indices", 0))
        << "if input sparse_feature exists, must have input sparse_indices.";
    const Shape& sparse_feature_shape = ctx->InputShape("sparse_feature", 0);
    const Shape& sparse_indices_shape = ctx->InputShape("sparse_indices", 0);
    CHECK_EQ_OR_RETURN(sparse_indices_shape.NumAxes(), 2)
        << "sparse_indices num_axes must be 2, but get " << sparse_indices_shape.NumAxes();
    CHECK_EQ_OR_RETURN(sparse_indices_shape.At(0), batch_size)
        << "get " << sparse_indices_shape.At(0) << " and " << batch_size;
    CHECK_EQ_OR_RETURN(sparse_feature_shape.At(sparse_feature_shape.NumAxes() - 1), vector_size)
        << "get " << sparse_feature_shape.At(sparse_feature_shape.NumAxes() - 1) << " and "
        << vector_size;
    features_concated_dim += sparse_indices_shape.At(1);
  }
  const bool self_interaction = ctx->Attr<bool>("self_interaction");
  const int32_t output_padding = ctx->Attr<int32_t>("output_padding");
  const int64_t interaction_dim = self_interaction
                                      ? features_concated_dim * (features_concated_dim + 1) / 2
                                      : features_concated_dim * (features_concated_dim - 1) / 2;
  int64_t out_dim = interaction_dim + output_padding;
  if (ctx->has_input("output_concat", 0)) {
    const Shape& output_concat_shape = ctx->InputShape("output_concat", 0);
    CHECK_EQ_OR_RETURN(output_concat_shape.NumAxes(), 2);
    CHECK_EQ_OR_RETURN(output_concat_shape.At(0), batch_size);
    out_dim += output_concat_shape.At(1);
  }
  ctx->SetOutputShape("out", 0, Shape({batch_size, out_dim}));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> FusedDotFeatureInteractionOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> FusedDotFeatureInteractionOp::GetSbp(user_op::SbpContext* ctx) {
  auto builder = ctx->NewBuilder().Split(ctx->inputs(), 0).Split(ctx->outputs(), 0);
  if (ctx->user_op_conf().has_input("num_valid_sparse_feature", 0)) {
    builder.Broadcast(user_op::OpArg("num_valid_sparse_feature", 0));
  }
  builder.Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> FusedDotFeatureInteractionOp::InferDataType(user_op::InferContext* ctx) {
  const int64_t feature_input_size = ctx->input_size("features");
  CHECK_GE_OR_RETURN(feature_input_size, 1);
  DataType first_feature_dtype = ctx->InputDType("features", 0);
  for (int64_t i = 1; i < feature_input_size; ++i) {
    CHECK_EQ_OR_RETURN(first_feature_dtype, ctx->InputDType("features", i))
        << "InferDataType Failed. Expected " << DataType_Name(ctx->InputDType("features", i))
        << ", but got " << DataType_Name(first_feature_dtype);
  }
  if (ctx->has_input("output_concat", 0)) {
    CHECK_EQ_OR_RETURN(first_feature_dtype, ctx->InputDType("output_concat", 0))
        << "InferDataType Failed. Expected " << DataType_Name(ctx->InputDType("output_concat", 0))
        << ", but got " << DataType_Name(first_feature_dtype);
  }
  if (ctx->has_input("sparse_feature", 0)) {
    CHECK_EQ_OR_RETURN(first_feature_dtype, ctx->InputDType("sparse_feature", 0))
        << "InferDataType Failed. Expected " << DataType_Name(ctx->InputDType("sparse_feature", 0))
        << ", but got " << DataType_Name(first_feature_dtype);
  }
  ctx->SetOutputDType("out", 0, first_feature_dtype);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> FusedDotFeatureInteractionGradOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  const Shape& dy_shape = ctx->InputShape("dy", 0);
  const int64_t batch_size = dy_shape.At(0);
  CHECK_EQ_OR_RETURN(ctx->output_size("features_grad"), ctx->input_size("features"))
      << "features_grad and features must have same size";
  for (int64_t i = 0; i < ctx->output_size("features_grad"); ++i) {
    ctx->SetOutputShape("features_grad", i, ctx->InputShape("features", i));
  }
  if (ctx->has_output("output_concat_grad", 0)) {
    const int32_t output_concat_grad_dim = ctx->Attr<int32_t>("output_concat_grad_dim");
    ctx->SetOutputShape("output_concat_grad", 0, Shape({batch_size, output_concat_grad_dim}));
  }
  if (ctx->has_output("sparse_feature_grad", 0)) {
    ctx->SetOutputShape("sparse_feature_grad", 0, ctx->InputShape("sparse_feature", 0));
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> FusedDotFeatureInteractionGradOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> FusedDotFeatureInteractionGradOp::GetSbp(user_op::SbpContext* ctx) {
  auto builder = ctx->NewBuilder().Split(ctx->inputs(), 0).Split(ctx->outputs(), 0);
  if (ctx->user_op_conf().has_input("num_valid_sparse_feature", 0)) {
    builder.Broadcast(user_op::OpArg("num_valid_sparse_feature", 0));
  }
  builder.Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> FusedDotFeatureInteractionGradOp::InferDataType(
    user_op::InferContext* ctx) {
  DataType dy_dtype = ctx->InputDType("dy", 0);
  for (int64_t i = 0; i < ctx->output_size("features_grad"); ++i) {
    ctx->SetOutputDType("features_grad", i, dy_dtype);
  }
  if (ctx->has_output("output_concat_grad", 0)) {
    ctx->SetOutputDType("output_concat_grad", 0, dy_dtype);
  }
  if (ctx->has_output("sparse_feature_grad", 0)) {
    ctx->SetOutputDType("sparse_feature_grad", 0, dy_dtype);
  }
  return Maybe<void>::Ok();
}

}  // namespace oneflow
