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

/*static*/ auto FusedBiasAddGeluOp::InferLogicalTensorDesc(user_op::InferContext* ctx)
    -> Maybe<void> {
  const auto& a_tensor_desc = ctx->InputTensorDesc("a", 0);
  const auto& b_tensor_desc = ctx->InputTensorDesc("b", 0);
  const auto bias_add_axis = ctx->Attr<int32_t>("axis");
  CHECK_EQ_OR_RETURN(b_tensor_desc.shape().NumAxes(), 1);
  CHECK_GE_OR_RETURN(bias_add_axis, 0);
  CHECK_LT_OR_RETURN(bias_add_axis, a_tensor_desc.shape().NumAxes());
  CHECK_EQ_OR_RETURN(a_tensor_desc.shape().At(bias_add_axis), b_tensor_desc.shape().At(0));
  *ctx->OutputShape("out", 0) = a_tensor_desc.shape();
  *ctx->OutputIsDynamic("out", 0) = a_tensor_desc.is_dynamic();
  return Maybe<void>::Ok();
}
/*static*/ auto FusedBiasAddGeluOp::InferPhysicalTensorDesc(user_op::InferContext* ctx)
    -> Maybe<void> {
  return FusedBiasAddGeluOp::InferLogicalTensorDesc(ctx);
}
/*static*/ auto FusedBiasAddGeluOp::InferDataType(user_op::InferContext* ctx) -> Maybe<void> {
  const auto& a_tensor_desc = ctx->InputTensorDesc("a", 0);
  *ctx->OutputDType("out", 0) = a_tensor_desc.data_type();
  return Maybe<void>::Ok();
}
/*static*/ auto FusedBiasAddGeluOp::GetSbp(user_op::SbpContext* ctx) -> Maybe<void> {
  const auto axis = ctx->Attr<int32_t>("axis");
  for (int64_t i = 0; i < ctx->LogicalTensorDesc4InputArgNameAndIndex("a", 0).shape().NumAxes();
       ++i) {
    if (i == axis) { continue; }
    ctx->NewBuilder()
        .Split(user_op::OpArg("a", 0), i)
        .Broadcast(user_op::OpArg("b", 0))
        .Split(ctx->outputs(), i)
        .Build();
  }
  ctx->NewBuilder()
      .Split(user_op::OpArg("b", 0), 0)
      .Split(user_op::OpArg("a", 0), axis)
      .Split(ctx->outputs(), axis)
      .Build();
  return Maybe<void>::Ok();
}
/*static*/ auto FusedBiasAddGeluGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx)
    -> Maybe<void> {
  const auto& a_tensor_desc = ctx->InputTensorDesc("a", 0);
  const auto& b_tensor_desc = ctx->InputTensorDesc("b", 0);
  const auto bias_add_axis = ctx->Attr<int32_t>("axis");
  CHECK_EQ_OR_RETURN(b_tensor_desc.shape().NumAxes(), 1);
  CHECK_GE_OR_RETURN(bias_add_axis, 0);
  CHECK_LT_OR_RETURN(bias_add_axis, a_tensor_desc.shape().NumAxes());
  CHECK_EQ_OR_RETURN(a_tensor_desc.shape().At(bias_add_axis), b_tensor_desc.shape().At(0));
  *ctx->OutputShape("dx", 0) = a_tensor_desc.shape();
  *ctx->OutputIsDynamic("dx", 0) = a_tensor_desc.is_dynamic();
  return Maybe<void>::Ok();
}

/*static*/ auto FusedBiasAddGeluGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx)
    -> Maybe<void> {
  return FusedBiasAddGeluGradOp::InferLogicalTensorDesc(ctx);
}
/*static*/ auto FusedBiasAddGeluGradOp::InferDataType(user_op::InferContext* ctx) -> Maybe<void> {
  const auto& a_tensor_desc = ctx->InputTensorDesc("a", 0);
  *ctx->OutputDType("dx", 0) = a_tensor_desc.data_type();
  return Maybe<void>::Ok();
}
/*static*/ auto FusedBiasAddGeluGradOp::GetSbp(user_op::SbpContext* ctx) -> Maybe<void> {
  const auto axis = ctx->Attr<int32_t>("axis");
  for (int64_t i = 0; i < ctx->LogicalTensorDesc4InputArgNameAndIndex("a", 0).shape().NumAxes();
       ++i) {
    if (i == axis) { continue; }
    ctx->NewBuilder()
        .Split(user_op::OpArg("a", 0), i)
        .Split(user_op::OpArg("dy", 0), i)
        .Broadcast(user_op::OpArg("b", 0))
        .Split(ctx->outputs(), i)
        .Build();
  }
  ctx->NewBuilder()
      .Split(user_op::OpArg("b", 0), 0)
      .Split(user_op::OpArg("a", 0), axis)
      .Split(user_op::OpArg("dy", 0), axis)
      .Split(ctx->outputs(), axis)
      .Build();
  return Maybe<void>::Ok();
}

REGISTER_USER_OP_GRAD("fused_bias_add_gelu")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                               user_op::AddOpFn AddOp) -> Maybe<void> {
      if (op.NeedGenGradTensor4OpInput("a", 0) || op.NeedGenGradTensor4OpInput("b", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_gelu_grad");
        user_op::UserOpConfWrapper bias_add_gelu_grad_op =
            builder.Op("fused_bias_add_gelu_grad")
                .Input("a", op.input("a", 0))
                .Input("b", op.input("b", 0))
                .Input("dy", op.GetGradTensorWithOpOutput("out", 0))
                .Attr("axis", op.attr<int32_t>("axis"))
                .Output("dx")
                .Build();
        AddOp(bias_add_gelu_grad_op);

        if (op.NeedGenGradTensor4OpInput("a", 0)) {
          op.BindGradTensorWithOpInput(bias_add_gelu_grad_op.output("dx", 0), "a", 0);
        }
        if (op.NeedGenGradTensor4OpInput("b", 0)) {
          const int64_t num_axes = op.TensorDesc4ArgNameAndIndex("a", 0).shape().NumAxes();
          const int32_t bias_add_axis = op.attr<int32_t>("axis");
          std::vector<int32_t> reduce_axes_vec;
          FOR_RANGE(int64_t, i, 0, num_axes) {
            if (i != bias_add_axis) { reduce_axes_vec.emplace_back(i); }
          }
          user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
          auto grad_op = builder.Op("reduce_sum")
                             .Input("input_tensor", bias_add_gelu_grad_op.output("dx", 0))
                             .Output("output_tensor")
                             .Attr("axis", reduce_axes_vec)
                             .Attr("keepdims", false)
                             .Build();
          AddOp(grad_op);
          op.BindGradTensorWithOpInput(grad_op.output("output_tensor", 0), "b", 0);
        }
      }
      return Maybe<void>::Ok();
    });

/*static*/ auto FusedBiasAddMaskScaleOp::InferLogicalTensorDesc(user_op::InferContext* ctx)
    -> Maybe<void> {
  const auto& a_tensor_desc = ctx->InputTensorDesc("a", 0);
  const auto& mask_tensor_desc = ctx->InputTensorDesc("mask", 0);
  const auto& b_tensor_desc = ctx->InputTensorDesc("b", 0);
  const auto bias_add_axis = ctx->Attr<int32_t>("axis");
  CHECK_EQ_OR_RETURN(b_tensor_desc.shape().NumAxes(), 1);
  CHECK_GE_OR_RETURN(bias_add_axis, 0);
  CHECK_LT_OR_RETURN(bias_add_axis, a_tensor_desc.shape().NumAxes());
  CHECK_EQ_OR_RETURN(a_tensor_desc.shape().At(bias_add_axis), b_tensor_desc.shape().At(0));
  CHECK_EQ_OR_RETURN(a_tensor_desc.shape(), mask_tensor_desc.shape());
  *ctx->OutputShape("out", 0) = a_tensor_desc.shape();
  *ctx->OutputIsDynamic("out", 0) = a_tensor_desc.is_dynamic();
  return Maybe<void>::Ok();
}
/*static*/ auto FusedBiasAddMaskScaleOp::InferPhysicalTensorDesc(user_op::InferContext* ctx)
    -> Maybe<void> {
  return FusedBiasAddMaskScaleOp::InferLogicalTensorDesc(ctx);
}
/*static*/ auto FusedBiasAddMaskScaleOp::InferDataType(user_op::InferContext* ctx) -> Maybe<void> {
  const auto& a_tensor_desc = ctx->InputTensorDesc("a", 0);
  *ctx->OutputDType("out", 0) = a_tensor_desc.data_type();
  return Maybe<void>::Ok();
}
/*static*/ auto FusedBiasAddMaskScaleOp::ModifyInputArg(
    const user_op::GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper&)
    -> Maybe<void> {
  user_op::InputArgModifier* mask_modifier = GetInputArgModifierFn("mask", 0);
  CHECK_OR_RETURN(mask_modifier != nullptr);
  mask_modifier->set_requires_grad(false);
  return Maybe<void>::Ok();
}
/*static*/ auto FusedBiasAddMaskScaleOp::GetSbp(user_op::SbpContext* ctx) -> Maybe<void> {
  const auto axis = ctx->Attr<int32_t>("axis");
  std::vector<user_op::OpArg> split_args;
  split_args.emplace_back("a", 0);
  split_args.emplace_back("mask", 0);
  split_args.emplace_back("out", 0);
  if (ctx->user_op_conf().has_input("_add_to_output", 0)) {
    split_args.emplace_back("_add_to_output", 0);
  }
  for (int64_t i = 0; i < ctx->LogicalTensorDesc4InputArgNameAndIndex("a", 0).shape().NumAxes();
       ++i) {
    if (i == axis) { continue; }
    ctx->NewBuilder().Split(split_args, i).Broadcast(user_op::OpArg("b", 0)).Build();
  }
  ctx->NewBuilder().Split(user_op::OpArg("b", 0), 0).Split(split_args, axis).Build();
  return Maybe<void>::Ok();
}

REGISTER_USER_OP_GRAD("fused_bias_add_mask_scale")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                               const user_op::AddOpFn& AddOp) -> Maybe<void> {
      if (op.NeedGenGradTensor4OpInput("a", 0) || op.NeedGenGradTensor4OpInput("b", 0)) {
        float scale = op.attr<float>("scale");
        if (scale != 1.0) {
          user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_dropout_grad");
          user_op::UserOpConfWrapper dropout_grad_op =
              builder.Op("dropout_grad")
                  .Input("dy", op.GetGradTensorWithOpOutput("out", 0))
                  .Input("mask", op.input("mask", 0))
                  .Output("dx")
                  .Attr("scale", scale)
                  .Build();
          AddOp(dropout_grad_op);

          if (op.NeedGenGradTensor4OpInput("a", 0)) {
            op.BindGradTensorWithOpInput(dropout_grad_op.output("dx", 0), "a", 0);
          }
          if (op.NeedGenGradTensor4OpInput("b", 0)) {
            const int64_t num_axes = op.TensorDesc4ArgNameAndIndex("a", 0).shape().NumAxes();
            const int32_t bias_add_axis = op.attr<int32_t>("axis");
            std::vector<int32_t> reduce_axes_vec;
            FOR_RANGE(int64_t, i, 0, num_axes) {
              if (i != bias_add_axis) { reduce_axes_vec.emplace_back(i); }
            }
            user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
            auto grad_op = builder.Op("reduce_sum")
                               .Input("input_tensor", dropout_grad_op.output("dx", 0))
                               .Output("output_tensor")
                               .Attr("axis", reduce_axes_vec)
                               .Attr("keepdims", false)
                               .Build();
            AddOp(grad_op);
            op.BindGradTensorWithOpInput(grad_op.output("output_tensor", 0), "b", 0);
          }
        } else {
          // When dropout_prob = 0.0, scale = 1.0, here we directly use out grad.
          if (op.NeedGenGradTensor4OpInput("a", 0)) {
            op.BindGradTensorWithOpInput(op.GetGradTensorWithOpOutput("out", 0), "a", 0);
          }
          if (op.NeedGenGradTensor4OpInput("b", 0)) {
            const int64_t num_axes = op.TensorDesc4ArgNameAndIndex("a", 0).shape().NumAxes();
            const int32_t bias_add_axis = op.attr<int32_t>("axis");
            std::vector<int32_t> reduce_axes_vec;
            FOR_RANGE(int64_t, i, 0, num_axes) {
              if (i != bias_add_axis) { reduce_axes_vec.emplace_back(i); }
            }
            user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
            auto grad_op = builder.Op("reduce_sum")
                               .Input("input_tensor", op.GetGradTensorWithOpOutput("out", 0))
                               .Output("output_tensor")
                               .Attr("axis", reduce_axes_vec)
                               .Attr("keepdims", false)
                               .Build();
            AddOp(grad_op);
            op.BindGradTensorWithOpInput(grad_op.output("output_tensor", 0), "b", 0);
          }
        }
      }
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
