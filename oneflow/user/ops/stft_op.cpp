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

const Stride InferOutputStride(const Shape& in_shape, bool onesided = true,
                               bool return_complex = false) {
  // TODO(yzm):support onesided and return_complex
  Stride out_stride(in_shape.NumAxes(), 0);
  if (in_shape.At(0) == 1) {
    out_stride = {2, 2 * (in_shape.At(2) / 2 + 1), 1};
  } else {
    auto last_dim_half_size = in_shape.At(2) / 2 + 1;
    out_stride = {last_dim_half_size * 2 * in_shape.At(1), 2, 2 * (in_shape.At(2) / 2 + 1), 1};
  }
  return out_stride;
}

const Shape InferOutputShape(const Shape& in_shape, bool onesided = true,
                             bool return_complex = false) {
  // TODO(yzm):support onesided and return_complex
  Shape out_shape;
  auto last_dim_half_size = in_shape.At(2) / 2 + 1;
  if (in_shape.At(0) == 1) {
    out_shape = {last_dim_half_size, in_shape.At(1), 2};
  } else {
    out_shape = {in_shape.At(0), last_dim_half_size, in_shape.At(1), 2};
  }
  return out_shape;
}

/* static */ Maybe<void> StftOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& in_shape = ctx->InputShape("input", 0);

  const Stride& out_stride = InferOutputStride(in_shape);
  const Shape& out_shape = InferOutputShape(in_shape);

  ctx->SetOutputStride("output", 0, out_stride);
  ctx->SetOutputShape("output", 0, out_shape);
  ctx->SetOutputIsDynamic("output", 0, ctx->InputIsDynamic("input", 0));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> StftOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> StftOp::GetSbp(user_op::SbpContext* ctx) { return Maybe<void>::Ok(); }

/* static */ Maybe<void> StftOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("output", 0, ctx->InputDType("input", 0));
  return Maybe<void>::Ok();
}

}  // namespace oneflow