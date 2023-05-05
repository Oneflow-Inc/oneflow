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
#include <string>
#include "oneflow/core/common/memory_format.pb.h"
#include "oneflow/core/common/throw.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

static Maybe<void> GetSbpIdentity(user_op::SbpContext* ctx, const Shape& shape) {
  for (int32_t i = 0; i < shape.size(); ++i) {
    ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
  }
  return Maybe<void>::Ok();
}

static Maybe<void> GetSbpContiguousToChannelsLast2d(user_op::SbpContext* ctx, const Shape& shape) {
  int ndim = shape.size();
  if (ndim <= 2) { return GetSbpIdentity(ctx, shape); }
  ctx->NewBuilder().Split(ctx->inputs(), 0).Split(ctx->outputs(), 0).Build();
  ctx->NewBuilder().Split(ctx->inputs(), 1).Split(ctx->outputs(), ndim - 1).Build();
  for (int32_t i = 0; i < ndim - 2; ++i) {
    ctx->NewBuilder().Split(ctx->inputs(), i + 2).Split(ctx->outputs(), i + 1).Build();
  }
  return Maybe<void>::Ok();
}

static Maybe<void> GetSbpChannelsLast2dToContiguous(user_op::SbpContext* ctx, const Shape& shape) {
  int ndim = shape.size();
  if (ndim <= 2) { return GetSbpIdentity(ctx, shape); }
  ctx->NewBuilder().Split(ctx->inputs(), 0).Split(ctx->outputs(), 0).Build();
  ctx->NewBuilder().Split(ctx->inputs(), ndim - 1).Split(ctx->outputs(), 1).Build();
  for (int32_t i = 0; i < ndim - 2; ++i) {
    ctx->NewBuilder().Split(ctx->inputs(), i + 1).Split(ctx->outputs(), i + 2).Build();
  }
  return Maybe<void>::Ok();
}

using GetSbpFunc = std::function<Maybe<void>(user_op::SbpContext* ctx, const Shape& shape)>;

static GetSbpFunc get_sbp_funcs[MemoryFormat_Max][MemoryFormat_Max] = {
    /*kContiguous->other*/ {GetSbpIdentity, GetSbpContiguousToChannelsLast2d},
    /*kChannelsLast->other*/ {GetSbpChannelsLast2dToContiguous, GetSbpIdentity},
};

static Maybe<void> GetConvertMemoryFormatSbp(user_op::SbpContext* ctx, const Shape& shape,
                                             MemoryFormat memory_format,
                                             MemoryFormat target_memory_format) {
  auto sbp_func = get_sbp_funcs[memory_format][target_memory_format];
  return sbp_func(ctx, shape);
}

static Stride ComputeChannelsLast2dStride(const Shape& shape) {
  DimVector stride(shape.size());
  switch (shape.size()) {
    case 4:
      stride[1] = 1;
      stride[3] = shape[1];
      stride[2] = stride[3] * shape[3];
      stride[0] = stride[2] * shape[2];
      return stride;
    case 3:
      stride[0] = 1;
      stride[2] = shape[0];
      stride[1] = stride[2] * shape[2];
      return stride;
    default: CHECK_OR_THROW(false) << "ChannelsLast2d doesn't support size " << shape.size();
  }
  return stride;
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
  const auto& memory_format_str = ctx->Attr<std::string>("memory_format");
  MemoryFormat memory_format = MemoryFormat::kContiguous;
  if (memory_format_str == "contiguous") {
    memory_format = MemoryFormat::kContiguous;
  } else if (memory_format_str == "channels_last") {
    memory_format = MemoryFormat::kChannelsLast;
  } else {
    CHECK_OR_THROW(false) << "unsupported memory format";
  }

  out_tensor_desc->set_is_dynamic(in_tensor_desc.is_dynamic());
  out_tensor_desc->set_shape(in_shape);
  out_tensor_desc->set_memory_format(memory_format);
  out_tensor_desc->set_stride(ComputeChannelsLast2dStride(in_tensor_desc.shape()));
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
