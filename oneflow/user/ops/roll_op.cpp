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

/*static*/ Maybe<void> RollOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& in_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
  const std::vector<int32_t>& dims = ctx->Attr<std::vector<int32_t>>("dims");

  CHECK_GT_OR_RETURN(dims.size(), 0)
      << Error::RuntimeError() << "The input list of dims doesn't allow to be empty";
  // NOTE(Liang Depeng): (dims.size == 1 && dims[0] == -1) means that user call flow.roll with
  // dims == None
  if (dims[0] != -1) {
    FOR_RANGE(int64_t, i, 0, in_tensor.shape().NumAxes()) {
      if (std::find(dims.begin(), dims.end(), i) == dims.end()) {
        ctx->NewBuilder()
            .Split(user_op::OpArg("in", 0), i)
            .Split(user_op::OpArg("out", 0), i)
            .Build();
      }
    }
  }

  ctx->NewBuilder()
      .PartialSum(user_op::OpArg("in", 0))
      .PartialSum(user_op::OpArg("out", 0))
      .Build();
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> RollOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& in_shape = ctx->InputShape("in", 0);
  *ctx->MutOutputShape("out", 0) = in_shape;
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> RollOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> RollOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("out", 0) = ctx->InputDType("in", 0);
  return Maybe<void>::Ok();
}

REGISTER_USER_OP_GRAD("roll").SetGenBackwardOpConfFn(
    [](const user_op::UserOpWrapper& op, const user_op::AddOpFn& AddOp) -> Maybe<void> {
      if (op.NeedGenGradTensor4OpInput("in", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        std::vector<int32_t> shifts = op.attr<std::vector<int32_t>>("shifts");

        // NOTE(Liang Depeng): reverse the roll process
        for (int i = 0; i < shifts.size(); ++i) { shifts[i] *= -1; }

        user_op::UserOpConfWrapper grad_op =
            builder.Op("roll")
                .Input("in", op.GetGradTensorWithOpOutput("out", 0))
                .Output("out")
                .Attr<std::vector<int32_t>>("shifts", shifts)
                .Attr<std::vector<int32_t>>("dims", op.attr<std::vector<int32_t>>("dims"))
                .Build();
        op.BindGradTensorWithOpInput(grad_op.output("out", 0), "in", 0);
        AddOp(grad_op);
      }
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
