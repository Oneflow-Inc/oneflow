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

/*static*/ Maybe<void> MedianOp::GetSbp(user_op::SbpContext* ctx) {
  const auto& in_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("input", 0);
  int64_t num_axes = in_tensor.shape().NumAxes();
  if (num_axes == 0) {
    ctx->NewBuilder().PartialSum(ctx->inputs()).PartialSum(ctx->outputs()).Build();
  }
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> MedianOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& ones_shape = {1};
  *ctx->OutputShape("output", 0) = ones_shape.RemoveOnes({0});
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> MedianOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> MedianOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("output", 0) = ctx->InputDType("input", 0);
  return Maybe<void>::Ok();
}

Maybe<void> GenerateBackwardOpConf4Median(const user_op::UserOpWrapper& op,
                                          const user_op::AddOpFn& AddOp) {
  if (op.NeedGenGradTensor4OpInput("input", 0)) {
    const auto& input = op.arg_tensor_desc("input", 0);
    std::vector<int32_t> axis(input.shape().NumAxes());
    std::iota(axis.begin(), axis.end(), 0);

    user_op::UserOpConfWrapperBuilder broadcast_out_builder(op.op_name() + "_grad_broadcast_out");
    user_op::UserOpConfWrapper broadcast_out_op = broadcast_out_builder.Op("broadcast_like")
                                                      .Input("x", op.output("output", 0))
                                                      .Input("like", op.input("input", 0))
                                                      .Attr("broadcast_axes", axis)
                                                      .Output("y")
                                                      .Build();
    AddOp(broadcast_out_op);

    user_op::UserOpConfWrapperBuilder broadcast_eq_builder(op.op_name() + "_grad_broadcast_eq");
    user_op::UserOpConfWrapper broadcast_eq_op = broadcast_eq_builder.Op("broadcast_equal")
                                                     .Input("x", op.input("input", 0))
                                                     .Input("y", broadcast_out_op.output("y", 0))
                                                     .Output("z")
                                                     .Build();
    AddOp(broadcast_eq_op);

    user_op::UserOpConfWrapperBuilder cast_mask_builder(op.op_name() + "_grad_cast_mask");
    user_op::UserOpConfWrapper cast_mask_op = cast_mask_builder.Op("cast_like")
                                                  .Input("in", broadcast_eq_op.output("z", 0))
                                                  .Input("dtype_like", op.input("input", 0))
                                                  .Output("out")
                                                  .Build();
    AddOp(cast_mask_op);

    user_op::UserOpConfWrapperBuilder reduce_sum_mask_builder(op.op_name()
                                                              + "_grad_reduce_sum_mask");
    user_op::UserOpConfWrapper reduce_sum_mask_op =
        reduce_sum_mask_builder.Op("reduce_sum")
            .Input("input_tensor", cast_mask_op.output("out", 0))
            .Output("output_tensor")
            .Attr("axis", axis)
            .Attr("keepdims", op.attr<bool>("keepdims"))
            .Build();
    AddOp(reduce_sum_mask_op);

    user_op::UserOpConfWrapperBuilder divide_count_builder(op.op_name() + "_grad_divide_count");
    user_op::UserOpConfWrapper divide_count_op =
        divide_count_builder.Op("broadcast_div")
            .Input("x", op.GetGradTensorWithOpOutput("output", 0))
            .Input("y", reduce_sum_mask_op.output("output_tensor", 0))
            .Output("z")
            .Build();
    AddOp(divide_count_op);

    user_op::UserOpConfWrapperBuilder broadcast_divided_dy_builder(op.op_name()
                                                                   + "_grad_broadcast_divided_dy");
    user_op::UserOpConfWrapper broadcast_divided_dy_op =
        broadcast_divided_dy_builder.Op("broadcast_like")
            .Input("x", divide_count_op.output("z", 0))
            .Input("like", op.input("input", 0))
            .Attr("broadcast_axis", axis)
            .Output("y")
            .Build();
    AddOp(broadcast_divided_dy_op);

    user_op::UserOpConfWrapperBuilder multiply_mask_builder(op.op_name() + "_grad_multiply_mask");
    user_op::UserOpConfWrapper multiply_mask_op =
        multiply_mask_builder.Op("multiply")
            .Input("x", broadcast_divided_dy_op.output("y", 0))
            .Input("y", cast_mask_op.output("out", 0))
            .Output("out")
            .Build();
    AddOp(multiply_mask_op);
    op.BindGradTensorWithOpInput(multiply_mask_op.output("out", 0), "input", 0);
  }
  return Maybe<void>::Ok();
}

REGISTER_USER_OP_GRAD("median").SetGenBackwardOpConfFn(GenerateBackwardOpConf4Median);

}  // namespace oneflow
