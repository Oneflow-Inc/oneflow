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

/*static*/ Maybe<void> SplitOp::GetSbp(user_op::SbpContext* ctx) {
  const int64_t in_num_axes =
      ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0).shape().NumAxes();
  int64_t axis = ctx->Attr<int64_t>("dim");
  if (axis < 0) { axis += in_num_axes; }
  CHECK_OR_RETURN(axis >= 0 && axis < in_num_axes);
  
  FOR_RANGE(int64_t, i, 0, in_num_axes) {
    if (i == axis) { continue; }
    ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
  }
  ctx->NewBuilder()
      .PartialSum(user_op::OpArg("x", 0))
      .PartialSum(ctx->outputs())
      .Build();
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> SplitOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const auto axis = ctx->Attr<int64_t>("dim");
  const auto& sections = ctx->Attr<std::vector<int64_t>>("sections");
  const user_op::TensorDesc& in_desc = ctx->InputTensorDesc("x", 0);
  const int64_t in_num_axes = ctx->InputTensorDesc("x", 0).shape().NumAxes();
  CHECK_LT_OR_RETURN(axis, in_num_axes);

  FOR_RANGE(int32_t, i, 0, sections.size()) {
    user_op::TensorDesc* out_i_desc = ctx->OutputTensorDesc("out", i);
    DimVector out_i_dim_vec = in_desc.shape().dim_vec();
    out_i_dim_vec[i] = sections[axis];
    *out_i_desc->mut_shape() = Shape(out_i_dim_vec);
    out_i_desc->set_is_dynamic(in_desc.is_dynamic());
  }
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> SplitOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> SplitOp::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& in_desc = ctx->InputTensorDesc("x", 0);
  FOR_RANGE(int32_t, i, 0, ctx->outputs().size()) {
    user_op::TensorDesc* out_i_desc = ctx->OutputTensorDesc("out", i);
    *out_i_desc->mut_data_type() = in_desc.data_type();
  }
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> SplitOp::CheckAttr(const user_op::UserOpDefWrapper&,
                                              const user_op::UserOpConfWrapper& op_conf) {
  CHECK_OR_RETURN(op_conf.output_size("out") >= 1);
  return Maybe<void>::Ok();
}

namespace {

// Todo: use concat
Maybe<void> GenGradOp(const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
  const int64_t axis = op.attr<int64_t>("dim");
  const int32_t out_size = op.output_size("out");
  int64_t max_dim_size = op.TensorDesc4ArgNameAndIndex("in", 0).shape().At(axis);
  if (op.NeedGenGradTensor4OpInput("x", 0)) {
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
      builder = builder.Input("x", out_diff_lbn);
    }
    user_op::UserOpConfWrapper grad_op =
        builder.Output("out").Attr("axis", axis).Attr("max_dim_size", max_dim_size).Build();

    op.BindGradTensorWithOpInput(grad_op.output("out", 0), "x", 0);
    AddOp(grad_op);
  }
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_USER_OP_GRAD("split").SetGenBackwardOpConfFn(GenGradOp);

}  // namespace oneflow
