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
#include "oneflow/user/ops/convert_memory_format_op.h"

#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

static Shape ComputeShapeIdentity(const Shape& shape) { return shape; }

Shape ComputeShapeContiguousToChannelsLast(const Shape& shape) {
  int ndim = shape.size();
  if (ndim <= 2) { return ComputeShapeIdentity(shape); }
  Shape target_shape(ndim);
  target_shape[0] = shape[0];
  target_shape[ndim - 1] = shape[1];
  for (int i = 0; i < ndim - 2; ++i) { target_shape[i + 1] = shape[i + 2]; }
  return target_shape;
}

Shape ComputeShapeChannelsLastToContiguous(const Shape& shape) {
  int ndim = shape.size();
  if (ndim <= 2) { return ComputeShapeIdentity(shape); }
  Shape target_shape(ndim);
  target_shape[0] = shape[0];
  target_shape[1] = shape[ndim - 1];
  for (int i = 0; i < ndim - 2; ++i) { target_shape[i + 2] = shape[i + 1]; }
  return target_shape;
}

static Maybe<void> GetSbpIdentity(user_op::SbpContext* ctx, const Shape& shape) {
  for (int32_t i = 0; i < shape.size(); ++i) {
    ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
  }
  return Maybe<void>::Ok();
}

static Maybe<void> GetSbpContiguousToChannelsLast(user_op::SbpContext* ctx, const Shape& shape) {
  int ndim = shape.size();
  if (ndim <= 2) { return GetSbpIdentity(ctx, shape); }
  ctx->NewBuilder().Split(ctx->inputs(), 0).Split(ctx->outputs(), 0).Build();
  ctx->NewBuilder().Split(ctx->inputs(), 1).Split(ctx->outputs(), ndim - 1).Build();
  for (int32_t i = 0; i < ndim - 2; ++i) {
    ctx->NewBuilder().Split(ctx->inputs(), i + 2).Split(ctx->outputs(), i + 1).Build();
  }
  return Maybe<void>::Ok();
}

static Maybe<void> GetSbpChannelsLastToContiguous(user_op::SbpContext* ctx, const Shape& shape) {
  int ndim = shape.size();
  if (ndim <= 2) { return GetSbpIdentity(ctx, shape); }
  ctx->NewBuilder().Split(ctx->inputs(), 0).Split(ctx->outputs(), 0).Build();
  ctx->NewBuilder().Split(ctx->inputs(), ndim - 1).Split(ctx->outputs(), 1).Build();
  for (int32_t i = 0; i < ndim - 2; ++i) {
    ctx->NewBuilder().Split(ctx->inputs(), i + 1).Split(ctx->outputs(), i + 2).Build();
  }
  return Maybe<void>::Ok();
}

using ComputeShapeFunc = std::function<Shape(const Shape&)>;
using GetSbpFunc = std::function<Maybe<void>(user_op::SbpContext* ctx, const Shape& shape)>;

static ComputeShapeFunc compute_shape_funcs[kMemoryFormatCount][kMemoryFormatCount] = {
    /*kContiguous->other*/ {ComputeShapeIdentity, ComputeShapeContiguousToChannelsLast},
    /*kChannelsLast->other*/ {ComputeShapeChannelsLastToContiguous, ComputeShapeIdentity},
};

static GetSbpFunc get_sbp_funcs[kMemoryFormatCount][kMemoryFormatCount] = {
    /*kContiguous->other*/ {GetSbpIdentity, GetSbpContiguousToChannelsLast},
    /*kChannelsLast->other*/ {GetSbpChannelsLastToContiguous, GetSbpIdentity},
};

Shape ComputeConvertMemoryFormatShape(const Shape& shape, MemoryFormat memory_format,
                                      MemoryFormat target_memory_format) {
  auto shape_func = compute_shape_funcs[memory_format][target_memory_format];
  return shape_func(shape);
}

static Maybe<void> GetConvertMemoryFormatSbp(user_op::SbpContext* ctx, const Shape& shape,
                                             MemoryFormat memory_format,
                                             MemoryFormat target_memory_format) {
  auto sbp_func = get_sbp_funcs[memory_format][target_memory_format];
  return sbp_func(ctx, shape);
}

/*static*/ Maybe<void> ConvertMemoryFormatOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& input_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
  const auto& memory_format = ctx->Attr<MemoryFormat>("memory_format");

  JUST(GetConvertMemoryFormatSbp(ctx, input_tensor.shape(), input_tensor.memory_format(),
                                 memory_format));
  ctx->NewBuilder().PartialSum(ctx->inputs()).PartialSum(ctx->outputs()).Build();
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> ConvertMemoryFormatOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& in_tensor_desc = ctx->InputTensorDesc("in", 0);
  user_op::TensorDesc* out_tensor_desc = ctx->MutOutputTensorDesc("out", 0);
  const Shape& in_shape = in_tensor_desc.shape();
  const auto& memory_format = ctx->Attr<MemoryFormat>("memory_format");

  out_tensor_desc->set_is_dynamic(in_tensor_desc.is_dynamic());
  out_tensor_desc->set_shape(
      ComputeConvertMemoryFormatShape(in_shape, in_tensor_desc.memory_format(), memory_format));
  out_tensor_desc->set_memory_format(memory_format);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> ConvertMemoryFormatOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/*static*/ Maybe<void> ConvertMemoryFormatOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, ctx->InputDType("in", 0));
  return Maybe<void>::Ok();
}

}  // namespace oneflow
