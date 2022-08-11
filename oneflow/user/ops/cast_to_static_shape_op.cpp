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

/* static */ Maybe<void> CastToStaticShapeOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& input_desc = ctx->InputTensorDesc("input", 0);
  user_op::TensorDesc* output_desc = ctx->MutOutputTensorDesc("output", 0);
  *output_desc->mut_shape() = input_desc.shape();
  output_desc->set_is_dynamic(false);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> CastToStaticShapeOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> CastToStaticShapeOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& input_desc = ctx->LogicalTensorDesc4InputArgNameAndIndex("input", 0);
  FOR_RANGE(int64_t, i, 0, input_desc.shape().NumAxes()) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("input", 0), i)
        .Split(user_op::OpArg("output", 0), i)
        .Build();
  }
  ctx->NewBuilder()
      .PartialSum(user_op::OpArg("input", 0))
      .PartialSum(user_op::OpArg("output", 0))
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> CastToStaticShapeOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->MutOutputDType("output", 0) = ctx->InputDType("input", 0);
  return Maybe<void>::Ok();
}

REGISTER_USER_OP_GRAD("cast_to_static_shape")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                               user_op::AddOpFn AddOp) -> Maybe<void> {
      if (op.NeedGenGradTensor4OpInput("input", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper identity_op =
            builder.Op("identity")
                .Input("in", op.GetGradTensorWithOpOutput("output", 0))
                .Output("out")
                .Build();
        op.BindGradTensorWithOpInput(identity_op.output("out", 0), "input", 0);
        AddOp(identity_op);
      }
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
