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
#include "oneflow/core/operator/reduce_sbp_util.h"
#include "oneflow/core/ndarray/binary_func.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

Maybe<void> VarOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& input_shape = ctx->InputShape("input", 0);
  const auto& reduce_axes = ctx->Attr<std::vector<int32_t>>("dim");
  CHECK_OR_RETURN(!reduce_axes.empty());
  const AxisVector reduce_axes_vec = {reduce_axes.begin(), reduce_axes.end()};
  const Shape& reduce_shape = CreateReducedShape(input_shape, reduce_axes_vec);
  const bool keepdim = ctx->Attr<bool>("keepdim");
  Shape output_shape;
  if (keepdim) {
    output_shape = reduce_shape;
  } else {
    output_shape = reduce_shape.RemoveOnes(reduce_axes_vec);
  }
  ctx->SetOutputShape("output", 0, output_shape);
  return Maybe<void>::Ok();
}

Maybe<void> VarOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

Maybe<void> VarOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("output", 0, ctx->InputDType("input", 0));
  return Maybe<void>::Ok();
}

Maybe<void> VarOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder().Broadcast(ctx->inputs()).Broadcast(ctx->outputs()).Build();
  const Shape& input_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("input", 0).shape();
  const int64_t ndim = input_shape.NumAxes();
  const std::vector<int32_t> axis = ctx->Attr<std::vector<int32_t>>("dim");
  const bool keepdim = ctx->Attr<bool>("keepdim");
  if (keepdim) {
    for (int i = 0; i < ndim; i++) {
      if (std::find(axis.begin(), axis.end(), i) == axis.end()) {
        ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
      }
    }
  } else {
    int offset = 0;
    for (int i = 0; i < ndim; i++) {
      if (std::find(axis.begin(), axis.end(), i) == axis.end()) {
        ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i - offset).Build();
      } else {
        offset += 1;
      }
    }
  }
  return Maybe<void>::Ok();
}

REGISTER_USER_OP_GRAD("variance")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                               const user_op::AddOpFn& AddOp) -> Maybe<void> {
      if (op.NeedGenGradTensor4OpInput("input", 0)) {
        const auto axis = op.attr<std::vector<int32_t>>("axis");
        const bool keepdim = op.attr<bool>("keepdim");
        user_op::UserOpConfWrapperBuilder reduce_sum_out_builder(op.op_name() + "_reduce_sum_out");
        auto reduce_sum_op = reduce_sum_out_builder.Op("reduce_sum")
                                 .Input("input_tensor", op.GetGradTensorWithOpOutput("output", 0))
                                 .Output("output_tensor")
                                 .Attr("axis", axis)
                                 .Attr("keepdim", keepdim)
                                 .Build();
        AddOp(reduce_sum_op);
        const auto& in_shape = op.TensorDesc4ArgNameAndIndex("input", 0).shape();
        size_t elem_cnt = 1;
        for (const auto& item : axis) { elem_cnt *= in_shape.At(item); }
        user_op::UserOpConfWrapperBuilder scalar_mul_builder(op.op_name() + "_scalar_mul_out_0");
        auto scalar_mul_op = scalar_mul_builder.Op("scalar_mul")
                                 .Input("in", reduce_sum_op.output("output_tensor", 0))
                                 .Output("out")
                                 .Attr("has_int_operand", 0)
                                 .Attr("int_operand", op.attr_or_default<int64_t>("int_operand", 0))
                                 .Attr("has_float_operand", 1)
                                 .Attr("float_operand", 1 / elem_cnt)
                                 .Build();
        AddOp(scalar_mul_op);
        user_op::UserOpConfWrapperBuilder sub_builder(op.op_name() + "_sub_out");
        auto sub_op = sub_builder.Op("broadcast_sub")
                          .Input("x", op.input("input", 0))
                          .Input("y", scalar_mul_op.output("out", 0))
                          .Output("z")
                          .Build();
        AddOp(sub_op);
        bool unbiased = op.attr<bool>("unbiased");
        const size_t correction = unbiased ? 1 : 0;
        scalar_mul_builder = user_op::UserOpConfWrapperBuilder(op.op_name() + "_scalar_mul_out_1");
        scalar_mul_op = scalar_mul_builder.Op("scalar_mul")
                            .Input("in", sub_op.output("z", 0))
                            .Output("out")
                            .Attr("has_int_operand", 0)
                            .Attr("int_operand", op.attr_or_default<int64_t>("int_operand", 0))
                            .Attr("has_float_operand", 1)
                            .Attr("float_operand", 2.0 / (elem_cnt - correction))
                            .Build();
        AddOp(scalar_mul_op);
        op.BindGradTensorWithOpInput(scalar_mul_op.output("out", 0), "input", 0);
      }
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
