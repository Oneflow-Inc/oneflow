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
#include "oneflow/core/job/nd_sbp_util.h"

namespace oneflow {

namespace {

Maybe<void> InferExpandOutputStride(const Shape& input_shape, const Stride& input_stride,
                                    const Shape& expand_shape, Stride* output_stride) {
  CHECK_EQ_OR_RETURN(input_shape.size(), input_stride.size());  // NOLINT(maybe-need-error-msg)
  size_t lpad = expand_shape.size() - input_shape.size();
  CHECK_GE_OR_RETURN(lpad, 0);  // NOLINT(maybe-need-error-msg)

  output_stride->resize(expand_shape.size(), 0);
  for (int i = expand_shape.size() - 1; i >= 0; --i) {
    int64_t dim = i < lpad ? 1 : input_shape[i - lpad];
    if (dim == expand_shape[i]) {
      if (i >= lpad) {
        output_stride->at(i) = input_stride[i - lpad];
      } else if (i < expand_shape.size() - 1) {
        output_stride->at(i) = output_stride->at(i + 1) * expand_shape[i + 1];
      }
    } else {
      CHECK_EQ_OR_RETURN(dim, 1);  // NOLINT(maybe-need-error-msg)
    }
  }
  // NOTE: expand op only can output contiguous stride,
  // because lazy don't support to_contiguous op for now
  *output_stride = Stride(expand_shape);
  return Maybe<void>::Ok();
}

}  // namespace

/* static */ Maybe<void> ExpandOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& input_shape = ctx->InputShape("in", 0);
  const Stride& input_stride = ctx->InputStride("in", 0);
  const Shape& expand_shape = ctx->Attr<Shape>("expand_shape");

  ctx->SetOutputShape("out", 0, expand_shape);

  Stride output_stride;
  JUST(InferExpandOutputStride(input_shape, input_stride, expand_shape, &output_stride));
  ctx->SetOutputStride("out", 0, output_stride);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> ExpandOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& input_shape = ctx->InputShape("in", 0);
  const Stride& input_stride = ctx->InputStride("in", 0);

  const auto& global_expand_shape = ctx->Attr<Shape>("expand_shape");
  const auto& output_sbp = ctx->NdSbp4ArgNameAndIndex("out", 0);
  const auto& device_mesh = *ctx->parallel_desc().hierarchy();
  const auto& rank = ctx->parallel_ctx().parallel_id();
  const auto local_view =
      GetTensorSliceView4ParallelId(device_mesh, output_sbp, global_expand_shape, rank);
  const auto& local_expand_shape = local_view.shape();
  ctx->SetOutputShape("out", 0, local_expand_shape);

  Stride output_stride;
  JUST(InferExpandOutputStride(input_shape, input_stride, local_expand_shape, &output_stride));
  ctx->SetOutputStride("out", 0, output_stride);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> ExpandOp::GetSbp(user_op::SbpContext* ctx) {
  const auto& global_in_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0).shape();
  const auto& global_expand_shape = ctx->Attr<Shape>("expand_shape");
  size_t lpad = global_expand_shape.size() - global_in_shape.size();
  CHECK_GE_OR_RETURN(lpad, 0);  // NOLINT(maybe-need-error-msg)

  for (size_t i = 0; i < global_in_shape.size(); ++i) {
    if (global_in_shape[i] == global_expand_shape[i + lpad]) {
      ctx->NewBuilder()
          .Split(user_op::OpArg("in", 0), i)
          .Split(user_op::OpArg("out", 0), i + lpad)
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
  ctx->SetOutputDType("out", 0, ctx->InputDType("in", 0));
  return Maybe<void>::Ok();
}

}  // namespace oneflow
