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
#include "oneflow/user/kernels/expand_kernel_utils.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

/* static */ Maybe<void> ExpandOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& input_shape = ctx->InputShape("in", 0);
  const std::vector<int32_t>& logical_expand_shape =
      ctx->Attr<std::vector<int32_t>>("logical_expand_shape");

  std::vector<int32_t> in_shape;
  in_shape.resize(input_shape.NumAxes());
  for (int i = 0; i < input_shape.NumAxes(); ++i) { in_shape[i] = input_shape.At(i); }

  std::vector<int32_t> out_shape;
  std::vector<int32_t> stride;
  CHECK_JUST(getOutShapeAndStrideForFp(in_shape, logical_expand_shape, out_shape, stride));

  Shape* output_shape = ctx->MutOutputShape("out", 0);
  DimVector dim_vec(out_shape.begin(), out_shape.end());
  *output_shape = Shape(dim_vec);

  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> ExpandOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> ExpandOp::GetSbp(user_op::SbpContext* ctx) {
  const std::vector<int32_t>& logical_in_shape =
      ctx->Attr<std::vector<int32_t>>("logical_in_shape");
  const std::vector<int32_t>& logical_expand_shape =
      ctx->Attr<std::vector<int32_t>>("logical_expand_shape");
  std::vector<int32_t> logical_out_shape;
  std::vector<int32_t> stride;
  CHECK_JUST(
      getOutShapeAndStride(logical_in_shape, logical_expand_shape, logical_out_shape, stride));

  int offset = logical_out_shape.size() - logical_in_shape.size();
  FOR_RANGE(int64_t, i, 0, logical_in_shape.size()) {
    if (logical_in_shape[i] == logical_out_shape[i + offset]) {
      ctx->NewBuilder()
          .Split(user_op::OpArg("in", 0), i)
          .Split(user_op::OpArg("out", 0), i + offset)
          .Build();
    }
  }

  ctx->NewBuilder()
      .PartialSum(user_op::OpArg("in", 0))
      .PartialSum(user_op::OpArg("out", 0))
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> ExpandOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("out", 0) = ctx->InputDType("in", 0);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> ExpandGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& input_shape = ctx->InputShape("in", 0);
  const std::vector<int32_t>& logical_out_shape =
      ctx->Attr<std::vector<int32_t>>("logical_out_shape");
  const std::vector<int32_t>& logical_expand_shape =
      ctx->Attr<std::vector<int32_t>>("logical_expand_shape");

  std::vector<int32_t> in_shape;
  in_shape.resize(input_shape.NumAxes());
  for (int i = 0; i < input_shape.NumAxes(); ++i) { in_shape[i] = input_shape.At(i); }
  std::vector<int32_t> out_shape;
  std::vector<int32_t> stride;
  CHECK_JUST(getOutShapeAndStrideForBp(logical_out_shape, logical_expand_shape, in_shape, out_shape,
                                       stride));

  Shape* output_shape = ctx->MutOutputShape("out", 0);
  DimVector dim_vec(out_shape.begin(), out_shape.end());
  *output_shape = Shape(dim_vec);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> ExpandGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> ExpandGradOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& input_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
  const std::vector<int32_t>& logical_out_shape =
      ctx->Attr<std::vector<int32_t>>("logical_out_shape");
  const std::vector<int32_t>& logical_expand_shape =
      ctx->Attr<std::vector<int32_t>>("logical_expand_shape");

  int offset = input_tensor.shape().NumAxes() - logical_out_shape.size();
  FOR_RANGE(int64_t, i, 0, logical_out_shape.size()) {
    if (logical_out_shape[i] == input_tensor.shape().At(i + offset)) {
      ctx->NewBuilder()
          .Split(user_op::OpArg("in", 0), i + offset)
          .Split(user_op::OpArg("out", 0), i)
          .Build();
    }
  }

  ctx->NewBuilder()
      .PartialSum(user_op::OpArg("in", 0))
      .PartialSum(user_op::OpArg("out", 0))
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> ExpandGradOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("out", 0) = ctx->InputDType("in", 0);
  return Maybe<void>::Ok();
}

REGISTER_USER_OP_GRAD("expand").SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                                                          user_op::AddOpFn AddOp) -> Maybe<void> {
  if (op.NeedGenGradTensor4OpInput("in", 0)) {
    user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
    user_op::UserOpConfWrapper grad_op =
        builder.Op("expand_grad")
            .Input("in", op.GetGradTensorWithOpOutput("out", 0))
            .Output("out")
            .Attr<std::vector<int32_t>>("logical_out_shape",
                                        op.attr<std::vector<int32_t>>("logical_in_shape"))
            .Attr<std::vector<int32_t>>("logical_expand_shape",
                                        op.attr<std::vector<int32_t>>("logical_expand_shape"))
            .Build();
    op.BindGradTensorWithOpInput(grad_op.output("out", 0), "in", 0);
    AddOp(grad_op);
  }
  return Maybe<void>::Ok();
});

}  // namespace oneflow
