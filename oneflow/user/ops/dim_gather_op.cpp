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
#include "oneflow/user/kernels/dim_gather_kernel_util.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

/* static */ Maybe<void> DimGatherOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& in = ctx->InputTensorDesc("input", 0);
  int64_t input_num_axes = in.shape().NumAxes();
  // For 0-dim tensor
  CHECK_GE_OR_RETURN(input_num_axes, 0);  // NOLINT
  CHECK_LE_OR_RETURN(input_num_axes, kDimGatherMaxDimCount);

  const user_op::TensorDesc& index = ctx->InputTensorDesc("index", 0);
  int64_t index_num_axes = index.shape().NumAxes();

  const int32_t dim = ctx->Attr<int32_t>("dim");
  // For 0-dim tensor
  CHECK_GE_OR_RETURN(dim, 0);
  CHECK_LE_OR_RETURN(dim, input_num_axes);                                         // NOLINT
  if (input_num_axes > 0) { CHECK_GE_OR_RETURN(input_num_axes, index_num_axes); }  // NOLINT

  CHECK_EQ_OR_RETURN(in.is_dynamic(), index.is_dynamic());

  user_op::TensorDesc* out = ctx->MutOutputTensorDesc("output", 0);
  out->set_shape(index.shape());

  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> DimGatherOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> DimGatherOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& index_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("index", 0);
  int64_t index_num_axes = index_tensor.shape().NumAxes();
  const int32_t dim = ctx->Attr<int32_t>("dim");

  FOR_RANGE(int64_t, i, 0, index_num_axes) {
    if (i != dim) {
      ctx->NewBuilder()
          .Split(user_op::OpArg("index", 0), i)
          .Split(user_op::OpArg("input", 0), i)
          .Split(user_op::OpArg("output", 0), i)
          .Build();
    } else if (i == dim) {
      ctx->NewBuilder()
          .Broadcast(user_op::OpArg("input", 0))
          .Split(user_op::OpArg("index", 0), i)
          .Split(user_op::OpArg("output", 0), i)
          .Build();
    }
  }
  ctx->NewBuilder()
      .PartialSum(user_op::OpArg("input", 0))
      .Broadcast(user_op::OpArg("index", 0))
      .PartialSum(user_op::OpArg("output", 0))
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> DimGatherOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper& conf) {
  user_op::InputArgModifier* indices_modifier = GetInputArgModifierFn("index", 0);
  CHECK_OR_RETURN(indices_modifier != nullptr);
  indices_modifier->set_requires_grad(false);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> DimGatherOp::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& index = ctx->InputTensorDesc("index", 0);
  CHECK_OR_RETURN(IsIndexDataType(index.data_type()));
  const user_op::TensorDesc& in = ctx->InputTensorDesc("input", 0);
  user_op::TensorDesc* out = ctx->MutOutputTensorDesc("output", 0);
  out->set_data_type(in.data_type());
  return Maybe<void>::Ok();
}

}  // namespace oneflow
