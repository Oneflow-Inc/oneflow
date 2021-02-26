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

namespace {

Maybe<void> InferTensorDesc(user_op::InferContext* ctx) {
  const auto axis = ctx->Attr<int64_t>("axis");
  const user_op::TensorDesc* in_desc = ctx->TensorDesc4ArgNameAndIndex("in", 0);
  int64_t dynamic_dim_size = 0;
  int64_t static_dim_size = 0;
  const int64_t in_num_axes = ctx->TensorDesc4ArgNameAndIndex("in", 0)->shape().NumAxes();
  const int64_t like_num_axes = ctx->TensorDesc4ArgNameAndIndex("like", 0)->shape().NumAxes();
  CHECK_LE_OR_RETURN(like_num_axes, in_num_axes);
  CHECK_LT_OR_RETURN(axis, like_num_axes);
  FOR_RANGE(int32_t, i, 0, ctx->outputs().size()) {
    const user_op::TensorDesc* like_i_desc = ctx->TensorDesc4ArgNameAndIndex("like", i);
    user_op::TensorDesc* out_i_desc = ctx->TensorDesc4ArgNameAndIndex("out", i);
    CHECK_EQ_OR_RETURN(like_i_desc->shape().NumAxes(), like_num_axes);
    FOR_RANGE(int64_t, j, 0, like_num_axes) {
      if (j == axis) {
        if (like_i_desc->is_dynamic()) {
          dynamic_dim_size += like_i_desc->shape().At(j);
        } else {
          static_dim_size += like_i_desc->shape().At(j);
        }
      } else {
        CHECK_EQ_OR_RETURN(in_desc->shape().At(j), like_i_desc->shape().At(j));
      }
    }
    DimVector out_i_dim_vec = like_i_desc->shape().dim_vec();
    FOR_RANGE(int64_t, j, like_num_axes, in_num_axes) {
      out_i_dim_vec.push_back(in_desc->shape().At(j));
    }
    *out_i_desc->mut_shape() = Shape(out_i_dim_vec);
    *out_i_desc->mut_data_type() = in_desc->data_type();
    out_i_desc->set_is_dynamic(like_i_desc->is_dynamic());
  }
  if (dynamic_dim_size == 0) {
    CHECK_EQ_OR_RETURN(static_dim_size, in_desc->shape().At(axis));
  } else {
    CHECK_LE_OR_RETURN(static_dim_size, in_desc->shape().At(axis));
  }
  return Maybe<void>::Ok();
}

void SetLikeArgModifier(user_op::GetInputArgModifier GetInputArgModifierFn,
                        const user_op::UserOpConfWrapper& user_op_conf) {
  FOR_RANGE(int32_t, i, 0, user_op_conf.input_size("like")) {
    user_op::InputArgModifier* like_modifier = GetInputArgModifierFn("like", i);
    CHECK_NOTNULL(like_modifier);
    like_modifier->set_requires_grad(false);
  }
}

Maybe<void> GetSbpSignature(user_op::SbpContext* ctx) {
  const auto axis = ctx->Attr<int64_t>("axis");
  const int64_t in_num_axes =
      ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0).shape().NumAxes();
  const int64_t like_num_axes =
      ctx->LogicalTensorDesc4InputArgNameAndIndex("like", 0).shape().NumAxes();
  FOR_RANGE(int64_t, i, 0, like_num_axes) {
    if (i == axis) { continue; }
    ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
  }
  std::vector<user_op::OpArg> like_arg_vec;
  const size_t like_arg_size = ctx->outputs().size();
  like_arg_vec.reserve(like_arg_size);
  FOR_RANGE(int32_t, i, 0, like_arg_size) { like_arg_vec.emplace_back("like", i); }
  FOR_RANGE(int64_t, i, like_num_axes, in_num_axes) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("in", 0), i)
        .Broadcast(like_arg_vec)
        .Split(ctx->outputs(), i)
        .Build();
    ctx->NewBuilder()
        .Split(user_op::OpArg("in", 0), i)
        .PartialSum(like_arg_vec)
        .Split(ctx->outputs(), i)
        .Build();
  }
  ctx->NewBuilder()
      .PartialSum(user_op::OpArg("in", 0))
      .PartialSum(like_arg_vec)
      .PartialSum(ctx->outputs())
      .Build();
  ctx->NewBuilder()
      .PartialSum(user_op::OpArg("in", 0))
      .Broadcast(like_arg_vec)
      .PartialSum(ctx->outputs())
      .Build();
  ctx->NewBuilder()
      .Broadcast(user_op::OpArg("in", 0))
      .PartialSum(like_arg_vec)
      .Broadcast(ctx->outputs())
      .Build();
  return Maybe<void>::Ok();
}

void GenGradOp(const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
  const int64_t axis = op.attr<int64_t>("axis");
  const int32_t out_size = op.output_size("out");
  int64_t max_dim_size = 0;
  FOR_RANGE(int32_t, i, 0, out_size) {
    max_dim_size += op.TensorDesc4ArgNameAndIndex("like", i).shape().At(axis);
  }
  if (op.NeedGenGradTensor4OpInput("in", 0)) {
    user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
    builder = builder.Op("concat");
    FOR_RANGE(int32_t, i, 0, out_size) {
      std::string out_diff_lbn;
      if (op.HasGradTensor4OpOutput("out", i)) {
        out_diff_lbn = op.GetGradTensorWithOpOutput("out", i);
      } else {
        auto zero_like_op = user_op::UserOpConfWrapperBuilder(op.op_name() + "_grad_zero_like_out_"
                                                              + std::to_string(i))
                                .Op("zero_like")
                                .Input("like", op.output("out", i))
                                .Output("out")
                                .Build();
        AddOp(zero_like_op);
        out_diff_lbn = zero_like_op.output("out", 0);
      }
      builder = builder.Input("in", out_diff_lbn);
    }
    user_op::UserOpConfWrapper grad_op =
        builder.Output("out").Attr("axis", axis).Attr("max_dim_size", max_dim_size).Build();

    op.BindGradTensorWithOpInput(grad_op.output("out", 0), "in", 0);
    AddOp(grad_op);
  }
}

}  // namespace

REGISTER_USER_OP("split_like")
    .Input("in")
    .InputWithMinimum("like", 2)
    .OutputWithMinimum("out", 2)
    .Attr<int64_t>("axis")
    .SetTensorDescInferFn(InferTensorDesc)
    .SetInputArgModifyFn(SetLikeArgModifier)
    .SetGetSbpFn(GetSbpSignature);

REGISTER_USER_OP_GRAD("split_like").SetGenBackwardOpConfFn(GenGradOp);

}  // namespace oneflow
