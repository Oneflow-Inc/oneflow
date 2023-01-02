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

namespace {

Maybe<void> InferMaskedFillTensorDesc(user_op::InferContext* ctx) {
  const Shape& mask_shape = ctx->InputShape("mask", 0);
  ctx->SetOutputShape("out", 0, mask_shape);
  return Maybe<void>::Ok();
}

Maybe<void> InferMaskedFillDataType(user_op::InferContext* ctx) {
  DataType mask_dtype = ctx->InputDType("mask", 0);
  CHECK_OR_RETURN(IsIntegralDataType(mask_dtype) || IsBoolDataType(mask_dtype));
  ctx->SetOutputDType("out", 0, ctx->InputDType("x", 0));
  return Maybe<void>::Ok();
}

Maybe<void> GetMaskedFillSbpSignatures(user_op::SbpContext* ctx) {
  const Shape& mask_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("mask", 0).shape();
  const Shape& x_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0).shape();
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
  return Maybe<void>::Ok();
}

Maybe<void> GetMaskedFillInputArgModify(const user_op::GetInputArgModifier& GetInputArgModifierFn,
                                        const user_op::UserOpConfWrapper&) {
  user_op::InputArgModifier* mask_arg_modifier = GetInputArgModifierFn("mask", 0);
  mask_arg_modifier->set_requires_grad(false);
  return Maybe<void>::Ok();
}

}  // namespace

/* static */ Maybe<void> MaskedFillOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return InferMaskedFillTensorDesc(ctx);
}

/*static*/ Maybe<void> MaskedFillOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> MaskedFillOp::GetSbp(user_op::SbpContext* ctx) {
  return GetMaskedFillSbpSignatures(ctx);
}

/* static */ Maybe<void> MaskedFillOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper& conf) {
  return GetMaskedFillInputArgModify(GetInputArgModifierFn, conf);
}

/* static */ Maybe<void> MaskedFillOp::InferDataType(user_op::InferContext* ctx) {
  return InferMaskedFillDataType(ctx);
}

}  // namespace oneflow
