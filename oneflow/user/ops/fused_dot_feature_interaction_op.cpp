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
    *ctx->OutputShape("out", 0) = Shape({batch_size, vector_size});
    return Maybe<void>::Ok();
  }
  const int64_t align_dim = 16;
  const int64_t concated_padded_dim =
      std::ceil(static_cast<float>(features_concated_dim) / static_cast<float>(align_dim))
      * align_dim;
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
  *ctx->OutputShape("out", 0) = Shape({batch_size, out_dim});
  if (ctx->has_output("padded_concated_features", 0)) {
    *ctx->OutputShape("padded_concated_features", 0) =
        Shape({batch_size, concated_padded_dim, vector_size});
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> FusedDotFeatureInteractionOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> FusedDotFeatureInteractionOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder().Split(ctx->inputs(), 0).Split(ctx->outputs(), 0).Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> FusedDotFeatureInteractionOp::InferDataType(user_op::InferContext* ctx) {
  const int64_t feature_input_size = ctx->input_size("features");
  CHECK_GE_OR_RETURN(feature_input_size, 1);
  const auto& first_feature_dtype = ctx->InputDType("features", 0);
  for (int64_t i = 1; i < feature_input_size; ++i) {
    CHECK_EQ_OR_RETURN(first_feature_dtype, ctx->InputDType("features", i));
  }
  if (ctx->has_input("output_concat", 0)) {
    CHECK_EQ_OR_RETURN(first_feature_dtype, ctx->InputDType("output_concat", 0));
  }
  *ctx->OutputDType("out", 0) = first_feature_dtype;
  if (ctx->has_output("padded_concated_features", 0)) {
    *ctx->OutputDType("padded_concated_features", 0) = first_feature_dtype;
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> FusedDotFeatureInteractionGradOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  const Shape& dy_shape = ctx->InputShape("dy", 0);
  if (ctx->has_input("padded_concated_features", 0)) {
    const Shape& padded_concated_features_shape = ctx->InputShape("padded_concated_features", 0);
    CHECK_EQ_OR_RETURN(dy_shape.NumAxes(), 2) << dy_shape.NumAxes();
    CHECK_EQ_OR_RETURN(padded_concated_features_shape.NumAxes(), 3)
        << padded_concated_features_shape.NumAxes();
  }
  const int64_t batch_size = dy_shape.At(0);
  CHECK_EQ_OR_RETURN(ctx->output_size("features_grad"), ctx->input_size("features_grad_like"));
  for (int64_t i = 0; i < ctx->output_size("features_grad"); ++i) {
    *ctx->OutputShape("features_grad", i) = ctx->InputShape("features_grad_like", i);
  }
  if (ctx->has_output("output_concat_grad", 0)) {
    const int32_t output_concat_grad_dim = ctx->Attr<int32_t>("output_concat_grad_dim");
    *ctx->OutputShape("output_concat_grad", 0) = Shape({batch_size, output_concat_grad_dim});
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> FusedDotFeatureInteractionGradOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> FusedDotFeatureInteractionGradOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder().Split(ctx->inputs(), 0).Split(ctx->outputs(), 0).Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> FusedDotFeatureInteractionGradOp::InferDataType(
    user_op::InferContext* ctx) {
  const auto& dy_dtype = ctx->InputDType("dy", 0);
  for (int64_t i = 0; i < ctx->output_size("features_grad"); ++i) {
    *ctx->OutputDType("features_grad", i) = dy_dtype;
  }
  if (ctx->has_output("output_concat_grad", 0)) {
    *ctx->OutputDType("output_concat_grad", 0) = dy_dtype;
  }
  return Maybe<void>::Ok();
}

REGISTER_USER_OP_GRAD("fused_dot_feature_interaction")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                               const user_op::AddOpFn& AddOp) -> Maybe<void> {
      user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
      builder.Op("fused_dot_feature_interaction_grad")
          .Input("dy", op.GetGradTensorWithOpOutput("out", 0))
          .Attr<bool>("self_interaction", op.attr<bool>("self_interaction"))
          .Attr<std::string>("pooling", op.attr<std::string>("pooling"));
      if (op.user_op_conf().has_output("padded_concated_features", 0)) {
        builder.Input("padded_concated_features", op.output("padded_concated_features", 0));
      }
      for (int64_t i = 0; i < op.input_size("features"); ++i) {
        builder.Input("features_grad_like", op.input("features", i));
      }
      if (op.user_op_conf().has_input("output_concat", 0)) {
        builder.Output("output_concat_grad")
            .Attr<int32_t>("output_concat_grad_dim",
                           op.TensorDesc4ArgNameAndIndex("output_concat", 0).shape().At(1));
      }
      builder.Output("features_grad", op.input_size("features"));
      auto grad_op = builder.Build();
      AddOp(grad_op);

      for (int64_t i = 0; i < op.input_size("features"); ++i) {
        if (op.NeedGenGradTensor4OpInput("features", i)) {
          op.BindGradTensorWithOpInput(grad_op.output("features_grad", i), "features", i);
        }
      }
      if (op.user_op_conf().has_input("output_concat", 0)) {
        if (op.NeedGenGradTensor4OpInput("output_concat", 0)) {
          op.BindGradTensorWithOpInput(grad_op.output("output_concat_grad", 0), "output_concat", 0);
        }
      }
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
