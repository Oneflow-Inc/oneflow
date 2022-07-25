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

/* static */ auto AsStridedOp::InferLogicalTensorDesc(user_op::InferContext* ctx) -> Maybe<void> {
  const auto& size = ctx->Attr<std::vector<int32_t>>("size");
  const auto& stride = ctx->Attr<std::vector<int32_t>>("stride");
  CHECK_EQ_OR_RETURN(size.size(), stride.size()) << "mismatch in length of strides and shape";
  DimVector out_vec;
  out_vec.insert(out_vec.end(), size.cbegin(), size.cend());
  user_op::TensorDesc* output_desc = ctx->OutputTensorDesc("output", 0);
  *output_desc->mut_shape() = Shape(out_vec);
  return Maybe<void>::Ok();
}
/*static*/ auto AsStridedOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) -> Maybe<void> {
  return AsStridedOp::InferLogicalTensorDesc(ctx);
}
/*static*/ auto AsStridedOp::GetSbp(user_op::SbpContext* ctx) -> Maybe<void> {
  return Maybe<void>::Ok();
}
/*static*/ auto AsStridedOp::InferDataType(user_op::InferContext* ctx) -> Maybe<void> {
  *ctx->MutOutputDType("output", 0) = ctx->InputDType("input", 0);
  return Maybe<void>::Ok();
}

/* static */ auto AsStridedGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx)
    -> Maybe<void> {
  const Shape& input_shape = ctx->InputShape("input", 0);
  user_op::TensorDesc* dx_desc = ctx->OutputTensorDesc("dx", 0);
  *dx_desc->mut_shape() = input_shape;
  return Maybe<void>::Ok();
}
/*static*/ auto AsStridedGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx)
    -> Maybe<void> {
  return AsStridedGradOp::InferLogicalTensorDesc(ctx);
}
/*static*/ auto AsStridedGradOp::GetSbp(user_op::SbpContext* ctx) -> Maybe<void> {
  return Maybe<void>::Ok();
}
/*static*/ auto AsStridedGradOp::InferDataType(user_op::InferContext* ctx) -> Maybe<void> {
  *ctx->MutOutputDType("dx", 0) = ctx->InputDType("input", 0);
  return Maybe<void>::Ok();
}

REGISTER_USER_OP_GRAD("as_strided")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                               const user_op::AddOpFn& AddOp) -> Maybe<void> {
      bool need_grad_weight = op.NeedGenGradTensor4OpInput("input", 0);
      if (need_grad_weight) {
        user_op::UserOpConfWrapperBuilder in_grad_builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper in_grad_op =
            in_grad_builder.Op("as_strided_grad")
                .Input("dy", op.GetGradTensorWithOpOutput("output", 0))
                .Input("input", op.input("input", 0))
                .Output("dx")
                .Attr("size", op.attr<std::vector<int32_t>>("size"))
                .Attr("stride", op.attr<std::vector<int32_t>>("stride"))
                .Attr("storage_offset", op.attr<int32_t>("storage_offset"))
                .Build();
        op.BindGradTensorWithOpInput(in_grad_op.output("dx", 0), "input", 0);
        AddOp(in_grad_op);
      }
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
