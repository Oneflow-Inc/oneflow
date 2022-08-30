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

Maybe<void> InferExpandOutputShapeAndStride(const Shape& input_shape, const Stride& input_stride,
                                            const Shape& expand_shape, Shape* output_shape,
                                            Stride* output_stride) {
  CHECK_EQ_OR_RETURN(input_shape.size(), input_stride.size());  // NOLINT(maybe-need-error-msg)
  size_t lpad = expand_shape.size() - input_shape.size();
  CHECK_GE_OR_RETURN(lpad, 0);  // NOLINT(maybe-need-error-msg)
  output_stride->resize(expand_shape.size());
  for (size_t i = 0; i < expand_shape.size(); ++i) {
    const auto& t_dim = expand_shape[i];
    if (i >= lpad) {
      const auto& dim = input_shape[i - lpad];
      const auto& stride = input_stride[i - lpad];
      if (dim == t_dim) {
        output_stride->at(i) = stride;
      } else {
        CHECK_EQ_OR_RETURN(dim, 1);  // NOLINT(maybe-need-error-msg)
        output_stride->at(i) = 0;
      }
    } else {
      output_stride->at(i) = 0;
    }
  }
  *output_shape = expand_shape;
  return Maybe<void>::Ok();
}

}  // namespace

/* static */ Maybe<void> ExpandOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& input_shape = ctx->InputShape("in", 0);
  const Stride& input_stride = ctx->InputStride("in", 0);
  Shape* output_shape = ctx->MutOutputShape("out", 0);
  Stride* output_stride = ctx->MutOutputStride("out", 0);
  const Shape& expand_shape = ctx->Attr<Shape>("expand_shape");
  return InferExpandOutputShapeAndStride(input_shape, input_stride, expand_shape, output_shape,
                                         output_stride);
}

/*static*/ Maybe<void> ExpandOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& input_shape = ctx->InputShape("in", 0);
  const Stride& input_stride = ctx->InputStride("in", 0);
  Shape* output_shape = ctx->MutOutputShape("out", 0);
  Stride* output_stride = ctx->MutOutputStride("out", 0);

  const auto& global_expand_shape = ctx->Attr<Shape>("expand_shape");
  const auto& output_sbp = ctx->NdSbp4ArgNameAndIndex("out", 0);
  const auto& device_mesh = *ctx->parallel_desc().hierarchy();
  const auto& rank = ctx->parallel_ctx().parallel_id();
  const auto local_view =
      GetTensorSliceView4ParallelId(device_mesh, output_sbp, global_expand_shape, rank);
  const auto& local_expand_shape = local_view.shape();

  return InferExpandOutputShapeAndStride(input_shape, input_stride, local_expand_shape,
                                         output_shape, output_stride);
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
  *ctx->MutOutputDType("out", 0) = ctx->InputDType("in", 0);
  return Maybe<void>::Ok();
}

}  // namespace oneflow
