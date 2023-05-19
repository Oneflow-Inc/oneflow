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
#include <cstdint>
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/op_generated.h"
namespace oneflow {

/* static */ Maybe<void> FftC2COp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& in_shape = ctx->InputShape("input", 0);
  Stride out_stride = Stride(in_shape);  // contiguous
  ctx->SetOutputShape("out", 0, in_shape);
  ctx->SetOutputStride("out", 0, out_stride);
  ctx->SetOutputIsDynamic("out", 0, ctx->InputIsDynamic("input", 0));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> FftC2COp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> FftC2COp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .PartialSum(user_op::OpArg("input", 0))
      .PartialSum(user_op::OpArg("out", 0))
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> FftC2COp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, ctx->InputDType("input", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> FftR2COp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& in_shape = ctx->InputShape("input", 0);
  const auto& dims = ctx->Attr<std::vector<int64_t>>("dims");
  bool onesided = ctx->Attr<bool>("onesided");

  Shape out_shape = in_shape;
  auto last_dim = dims.back();
  if (onesided) { out_shape[last_dim] = out_shape[last_dim] / 2 + 1; }
  Stride out_stride = Stride(out_shape);
  ctx->SetOutputShape("out", 0, out_shape);
  ctx->SetOutputStride("out", 0, out_stride);
  ctx->SetOutputIsDynamic("out", 0, ctx->InputIsDynamic("input", 0));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> FftR2COp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> FftR2COp::GetSbp(user_op::SbpContext* ctx) {
  // TO-DO : Validate sbp
  ctx->NewBuilder().PartialSum(ctx->inputs()).PartialSum(ctx->outputs()).Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> FftR2COp::InferDataType(user_op::InferContext* ctx) {
  const DataType& input_type = ctx->InputDType("input", 0);
  switch (input_type) {
    case (kFloat): ctx->SetOutputDType("out", 0, kComplex64); break;
    case (kDouble): ctx->SetOutputDType("out", 0, kComplex128); break;
    default: CHECK_OR_RETURN(false) << "RuntimeError: dtype can't be handled";
  }

  return Maybe<void>::Ok();
}

/* static */ Maybe<void> FftC2ROp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& in_shape = ctx->InputShape("input", 0);

  const auto& dims = ctx->Attr<std::vector<int64_t>>("dims");
  int64_t last_dim_size = ctx->Attr<int64_t>("last_dim_size");

  Shape out_shape = in_shape;
  out_shape[dims.back()] = last_dim_size;
  Stride out_stride = Stride(out_shape);
  ctx->SetOutputShape("out", 0, out_shape);
  ctx->SetOutputStride("out", 0, out_stride);
  ctx->SetOutputIsDynamic("out", 0, ctx->InputIsDynamic("input", 0));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> FftC2ROp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> FftC2ROp::GetSbp(user_op::SbpContext* ctx) {
  // TO-DO : Validate sbp
  ctx->NewBuilder().PartialSum(ctx->inputs()).PartialSum(ctx->outputs()).Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> FftC2ROp::InferDataType(user_op::InferContext* ctx) {
  const DataType& input_type = ctx->InputDType("input", 0);
  switch (input_type) {
    case (kComplex64): ctx->SetOutputDType("out", 0, kFloat); break;
    case (kComplex128): ctx->SetOutputDType("out", 0, kDouble); break;
    default: CHECK_OR_RETURN(false) << "RuntimeError: dtype can't be handled";
  }

  return Maybe<void>::Ok();
}

}  // namespace oneflow