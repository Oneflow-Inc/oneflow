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

/*static*/ Maybe<void> MedianWithIndicesOp::GetSbp(user_op::SbpContext* ctx) {
  const auto& in_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("input", 0);
  int64_t num_axes = in_tensor.shape().NumAxes();
  FOR_RANGE(int64_t, i, 0, num_axes - 1) {
    ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
  }
  if (num_axes == 0) {
    ctx->NewBuilder().PartialSum(ctx->inputs()).PartialSum(ctx->outputs()).Build();
  }
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> MedianWithIndicesOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& input_shape = ctx->InputShape("input", 0);
  Shape* values_shape = ctx->MutOutputShape("values", 0);
  Shape* indices_shape = ctx->MutOutputShape("indices", 0);
  const Shape& reduce_shape = CreateReducedShape(input_shape, {-1});
  *values_shape = reduce_shape.RemoveOnes({-1});
  *indices_shape = reduce_shape.RemoveOnes({-1});
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> MedianWithIndicesOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> MedianWithIndicesOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->MutOutputDType("values", 0) = ctx->InputDType("input", 0);
  *ctx->MutOutputDType("indices", 0) = DataType::kInt64;
  return Maybe<void>::Ok();
}

Maybe<void> GenerateBackwardOpConf4MedianWithIndices(const user_op::UserOpWrapper& op,
                                                     const user_op::AddOpFn& AddOp) {
  if (op.NeedGenGradTensor4OpInput("input", 0)) {
    const auto& input = op.arg_tensor_desc("input", 0);
    user_op::UserOpConfWrapperBuilder expand_indices_builder(op.op_name() + "_grad_expand_indices");
    user_op::UserOpConfWrapper expand_indices_op = expand_indices_builder.Op("expand_dims")
                                                       .Input("in", op.output("indices", 0))
                                                       .Attr("axis", -1)
                                                       .Output("out")
                                                       .Build();
    AddOp(expand_indices_op);

    user_op::UserOpConfWrapperBuilder expand_dout_builder(op.op_name() + "_grad_expand_dout");
    user_op::UserOpConfWrapper expand_dout_op =
        expand_dout_builder.Op("expand_dims")
            .Input("in", op.GetGradTensorWithOpOutput("output", 0))
            .Attr("axis", -1)
            .Output("out")
            .Build();
    AddOp(expand_dout_op);

    bool is_integral = IsIntegralDataType(input.data_type());
    user_op::UserOpConfWrapperBuilder zeros_builder(op.op_name() + "_grad_zeros");
    user_op::UserOpConfWrapper zeros_op = zeros_builder.Op("constant")
                                              .Attr("shape", input.shape())
                                              .Attr("dtype", input.data_type())
                                              .Attr("is_floating_value", is_integral ? false : true)
                                              .Output("out")
                                              .Build();
    AddOp(zeros_op);

    user_op::UserOpConfWrapperBuilder dim_scatter_update_builder(op.op_name()
                                                                 + "_grad_dim_scatter_update");
    user_op::UserOpConfWrapper dim_scatter_update_op =
        dim_scatter_update_builder.Op("dim_scatter_update")
            .Input("input", zeros_op.output("out", 0))
            .Input("index", expand_indices_op.output("out", 0))
            .Input("src", expand_dout_op.output("out", 0))
            .Attr("dim", input.shape().NumAxes())
            .Output("output")
            .Build();
    AddOp(dim_scatter_update_op);
    op.BindGradTensorWithOpInput(dim_scatter_update_op.output("output", 0), "input", 0);
  }
  return Maybe<void>::Ok();
}

REGISTER_USER_OP_GRAD("median_with_indices")
    .SetGenBackwardOpConfFn(GenerateBackwardOpConf4MedianWithIndices);

}  // namespace oneflow
