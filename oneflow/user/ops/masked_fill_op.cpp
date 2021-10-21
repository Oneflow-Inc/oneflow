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

Maybe<void> InferMaskedFillTensorDesc(user_op::InferContext* ctx) {
  const Shape& mask_shape = ctx->InputShape("mask", 0);
  const Shape& x_shape = ctx->InputShape("x", 0);
  if (mask_shape == x_shape) {
    *ctx->OutputShape("out", 0) = mask_shape;
  } else {
    Shape max_shape = Shape::Ones(std::max(x_shape.NumAxes(), mask_shape.NumAxes()));
    const Shape& x_extend_shape = CreateLeftExtendedShape(ShapeView(x_shape), max_shape.NumAxes());
    const Shape& mask_extend_shape =
        CreateLeftExtendedShape(ShapeView(mask_shape), max_shape.NumAxes());
    FOR_RANGE(int64_t, i, 0, max_shape.NumAxes()) {
      max_shape.Set(i, std::max(x_extend_shape.At(i), mask_extend_shape.At(i)));
    }
    *ctx->OutputShape("out", 0) = max_shape;
  }
  return Maybe<void>::Ok();
}

Maybe<void> InferMaskedFillDataType(user_op::InferContext* ctx) {
  const DataType& mask_dtype = ctx->InputDType("mask", 0);
  CHECK_OR_RETURN(IsIntegralDataType(mask_dtype));
  *ctx->OutputDType("out", 0) = ctx->InputDType("x", 0);
  return Maybe<void>::Ok();
}

Maybe<void> GetMaskedFillSbpSignatures(user_op::SbpContext* ctx) {
  const Shape& mask_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("mask", 0).shape();
  const Shape& x_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0).shape();
  if (mask_shape.NumAxes() < x_shape.NumAxes()) {
    FOR_RANGE(int64_t, i, 0, x_shape.NumAxes() - mask_shape.NumAxes()) {
      ctx->NewBuilder()
          .Broadcast(user_op::OpArg("mask", 0))
          .Split(user_op::OpArg("x", 0), i)
          .Split(user_op::OpArg("out", 0), i)
          .Build();
    }
    FOR_RANGE(int64_t, i, 0, mask_shape.NumAxes()) {
      ctx->NewBuilder()
          .Split(user_op::OpArg("mask", 0), mask_shape.NumAxes() - 1 - i)
          .Split(user_op::OpArg("x", 0), x_shape.NumAxes() - 1 - i)
          .Split(ctx->outputs(), x_shape.NumAxes() - 1 - i)
          .Build();
    }
  } else if (mask_shape.NumAxes() > x_shape.NumAxes()) {
    FOR_RANGE(int64_t, i, 0, mask_shape.NumAxes() - x_shape.NumAxes()) {
      ctx->NewBuilder()
          .Split(user_op::OpArg("mask", 0), i)
          .Broadcast(user_op::OpArg("x", 0))
          .Split(user_op::OpArg("out", 0), i)
          .Build();
    }
    FOR_RANGE(int64_t, i, 0, x_shape.NumAxes()) {
      ctx->NewBuilder()
          .Split(user_op::OpArg("mask", 0), mask_shape.NumAxes() - 1 - i)
          .Split(user_op::OpArg("x", 0), x_shape.NumAxes() - 1 - i)
          .Split(ctx->outputs(), mask_shape.NumAxes() - 1 - i)
          .Build();
    }
  } else {
    FOR_RANGE(int64_t, i, 0, mask_shape.NumAxes()) {
      if (mask_shape.At(i) == 1 && x_shape.At(i) == 1) { continue; }
      if (mask_shape.At(i) == x_shape.At(i)) {
        ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
      } else if (mask_shape.At(i) == 1) {
        ctx->NewBuilder()
            .Broadcast(user_op::OpArg("mask", 0))
            .Split(user_op::OpArg("x", 0), i)
            .Split(ctx->outputs(), i)
            .Build();
      } else if (x_shape.At(i) == 1) {
        ctx->NewBuilder()
            .Split(user_op::OpArg("mask", 0), i)
            .Broadcast(user_op::OpArg("x", 0))
            .Split(ctx->outputs(), i)
            .Build();
      } else {
        UNIMPLEMENTED();
      }
    }
  }
  ctx->NewBuilder()
      .Broadcast(user_op::OpArg("mask", 0))
      .PartialSum(user_op::OpArg("x", 0))
      .PartialSum(user_op::OpArg("out", 0))
      .Build();
  return Maybe<void>::Ok();
}

Maybe<void> GetMaskedFillInputArgModify(user_op::GetInputArgModifier GetInputArgModifierFn,
                                        const user_op::UserOpConfWrapper&) {
  user_op::InputArgModifier* mask_arg_modifier = GetInputArgModifierFn("mask", 0);
  mask_arg_modifier->set_requires_grad(false);
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_USER_OP("masked_fill")
    .Input("x")
    .Input("mask")
    .Output("out")
    .Attr<bool>("has_int_operand")
    .Attr<bool>("has_float_operand")
    .Attr<int64_t>("int_operand")
    .Attr<double>("float_operand")
    .SetTensorDescInferFn(InferMaskedFillTensorDesc)
    .SetInputArgModifyFn(GetMaskedFillInputArgModify)
    .SetDataTypeInferFn(InferMaskedFillDataType)
    .SetGetSbpFn(GetMaskedFillSbpSignatures);

}  // namespace oneflow
