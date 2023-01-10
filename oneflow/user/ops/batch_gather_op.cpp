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

/* static */ Maybe<void> BatchGatherOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& in = ctx->InputTensorDesc("in", 0);
  CHECK_GT_OR_RETURN(in.shape().NumAxes(), 0)
      << Error::RuntimeError() << "The dimension of the input tensor should be greater than zero, "
      << "but got " << in.shape().NumAxes();
  const user_op::TensorDesc& indices = ctx->InputTensorDesc("indices", 0);
  CHECK_GT_OR_RETURN(indices.shape().NumAxes(), 0)
      << Error::RuntimeError()
      << "The dimension of the indices tensor should be greater than zero, "
      << "but got " << indices.shape().NumAxes();
  user_op::TensorDesc* out = ctx->MutOutputTensorDesc("out", 0);
  CHECK_LE_OR_RETURN(indices.shape().dim_vec().size(), in.shape().dim_vec().size())
      << Error::RuntimeError()
      << "The dimension of the input tensor should be greater than or equal to the dimension of "
         "the indices tensor, "
      << "but found that the dimension of the input tensor is " << in.shape().dim_vec().size()
      << ", and the dimension of the indices tensor is " << indices.shape().dim_vec().size();
  FOR_RANGE(int64_t, i, 0, indices.shape().dim_vec().size() - 1) {
    if (in.is_dynamic() && indices.is_dynamic() == false) {
      CHECK_GE_OR_RETURN(indices.shape().dim_vec().at(i), in.shape().dim_vec().at(i))
          << Error::RuntimeError()
          << "The size of indices tensor should be greater than or equal to the "
             "size of input tensor "
          << " at dimension " << i
          << " when the input tensor is dynamic and the indices tensor is not dynamic";
    } else if (in.is_dynamic() == false && indices.is_dynamic()) {
      LOG(FATAL)
          << "The indices tensor is not allowed to be dynamic when the input tensor is not dynamic";
    } else {
      CHECK_EQ_OR_RETURN(indices.shape().dim_vec().at(i), in.shape().dim_vec().at(i))
          << Error::RuntimeError()
          << "The size of indices tensor must match the size of input tensor"
          << " at dimension " << i << " when two tensors are both dynamic or neither";
    }
  }

  DimVector dim_vec(in.shape().dim_vec());
  dim_vec.at(indices.shape().NumAxes() - 1) = indices.shape().dim_vec().back();
  out->set_shape(Shape(dim_vec));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> BatchGatherOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> BatchGatherOp::GetSbp(user_op::SbpContext* ctx) {
  const int64_t indices_num_axes =
      ctx->LogicalTensorDesc4InputArgNameAndIndex("indices", 0).shape().NumAxes();
  if (indices_num_axes > 1) {
    FOR_RANGE(int64_t, i, 0, indices_num_axes - 1) {
      ctx->NewBuilder()
          .Split(user_op::OpArg("indices", 0), i)
          .Split(user_op::OpArg("in", 0), i)
          .Split(user_op::OpArg("out", 0), i)
          .Build();
    }
  }
  ctx->NewBuilder()
      .Broadcast(user_op::OpArg("indices", 0))
      .PartialSum(user_op::OpArg("in", 0))
      .PartialSum(user_op::OpArg("out", 0))
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> BatchGatherOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper& conf) {
  user_op::InputArgModifier* indices_modifier = GetInputArgModifierFn("indices", 0);
  CHECK_OR_RETURN(indices_modifier != nullptr);  // NOLINT(maybe-need-error-msg)
  indices_modifier->set_requires_grad(false);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> BatchGatherOp::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& indices = ctx->InputTensorDesc("indices", 0);
  CHECK_OR_RETURN(IsIndexDataType(indices.data_type()))
      << Error::TypeError() << "The dtype of the indices tensor must be int32 or int64";
  const user_op::TensorDesc& in = ctx->InputTensorDesc("in", 0);
  user_op::TensorDesc* out = ctx->MutOutputTensorDesc("out", 0);
  out->set_data_type(in.data_type());
  return Maybe<void>::Ok();
}

}  // namespace oneflow
