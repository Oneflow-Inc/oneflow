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

/* static */ Maybe<void> Chunk3Op::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& in = ctx->InputTensorDesc("in", 0);

  user_op::TensorDesc* out1 = ctx->MutOutputTensorDesc("out1", 0);
  user_op::TensorDesc* out2 = ctx->MutOutputTensorDesc("out2", 0);
  user_op::TensorDesc* out3 = ctx->MutOutputTensorDesc("out3", 0);

  const int64_t final_dim = in.shape().NumAxes() - 1;
  DimVector dim_vec;
  dim_vec.insert(dim_vec.end(), in.shape().dim_vec().cbegin(),
                 in.shape().dim_vec().cbegin() + final_dim);
  dim_vec.push_back(in.shape().At(final_dim) / 3);
  out1->set_shape(Shape(dim_vec));
  out2->set_shape(Shape(dim_vec));
  out3->set_shape(Shape(dim_vec));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> Chunk3Op::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> Chunk3Op::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& in_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
  FOR_RANGE(int64_t, i, 0, in_tensor.shape().NumAxes()) {
    if (i != in_tensor.shape().NumAxes() - 1) {
      ctx->NewBuilder()
          .Split(user_op::OpArg("in", 0), i)
          .Split(user_op::OpArg("out1", 0), i)
          .Split(user_op::OpArg("out2", 0), i)
          .Split(user_op::OpArg("out3", 0), i)
          .Build();
    }
  }
  ctx->NewBuilder()
      .PartialSum(user_op::OpArg("in", 0))
      .PartialSum(user_op::OpArg("out1", 0))
      .PartialSum(user_op::OpArg("out2", 0))
      .PartialSum(user_op::OpArg("out3", 0))
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> Chunk3Op::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& in = ctx->InputTensorDesc("in", 0);
  user_op::TensorDesc* out1 = ctx->MutOutputTensorDesc("out1", 0);
  out1->set_data_type(in.data_type());
  user_op::TensorDesc* out2 = ctx->MutOutputTensorDesc("out2", 0);
  out2->set_data_type(in.data_type());
  user_op::TensorDesc* out3 = ctx->MutOutputTensorDesc("out3", 0);
  out3->set_data_type(in.data_type());
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> Chunk3GradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& dout1 = ctx->InputTensorDesc("dout1", 0);
  const user_op::TensorDesc& dout2 = ctx->InputTensorDesc("dout2", 0);
  const user_op::TensorDesc& dout3 = ctx->InputTensorDesc("dout3", 0);
  user_op::TensorDesc* dx = ctx->MutOutputTensorDesc("dx", 0);

  DimVector dim_vec;
  const int64_t final_dim = dout1.shape().NumAxes() - 1;
  dim_vec.insert(dim_vec.end(), dout1.shape().dim_vec().cbegin(),
                 dout1.shape().dim_vec().cbegin() + final_dim);
  dim_vec.push_back(dout1.shape().At(final_dim) + dout2.shape().At(final_dim)
                    + dout3.shape().At(final_dim));

  dx->set_shape(Shape(dim_vec));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> Chunk3GradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> Chunk3GradOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& dout1_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("dout1", 0);
  FOR_RANGE(int64_t, i, 0, dout1_tensor.shape().NumAxes()) {
    if (i != dout1_tensor.shape().NumAxes() - 1) {
      ctx->NewBuilder()
          .Split(user_op::OpArg("dout1", 0), i)
          .Split(user_op::OpArg("dout2", 0), i)
          .Split(user_op::OpArg("dout3", 0), i)
          .Split(user_op::OpArg("dx", 0), i)
          .Build();
    }
  }
  ctx->NewBuilder()
      .PartialSum(user_op::OpArg("dout1", 0))
      .PartialSum(user_op::OpArg("dout2", 0))
      .PartialSum(user_op::OpArg("dout3", 0))
      .PartialSum(user_op::OpArg("dx", 0))
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> Chunk3GradOp::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& dout1 = ctx->InputTensorDesc("dout1", 0);
  user_op::TensorDesc* dx = ctx->MutOutputTensorDesc("dx", 0);
  dx->set_data_type(dout1.data_type());
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> Chunk3GradOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper& conf) {
  user_op::InputArgModifier* dout1_modifier = GetInputArgModifierFn("dout1", 0);
  CHECK_NOTNULL_OR_RETURN(dout1_modifier);
  dout1_modifier->set_requires_grad(false);
  user_op::InputArgModifier* dout2_modifier = GetInputArgModifierFn("dout2", 0);
  CHECK_NOTNULL_OR_RETURN(dout2_modifier);
  dout2_modifier->set_requires_grad(false);
  user_op::InputArgModifier* dout3_modifier = GetInputArgModifierFn("dout3", 0);
  CHECK_NOTNULL_OR_RETURN(dout3_modifier);
  dout3_modifier->set_requires_grad(false);
  return Maybe<void>::Ok();
}

}  // namespace oneflow