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

/*static*/ Maybe<void> SoftmaxCrossEntropyOp::GetSbp(user_op::SbpContext* ctx) {
  // ctx->LogicalTensorDesc4InputArgNameAndIndex("out", 0) is not initialized here
  const auto num_out_axes =
      ctx->LogicalTensorDesc4InputArgNameAndIndex("prediction", 0).shape().NumAxes() - 1;
  FOR_RANGE(int64_t, i, 0, num_out_axes) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("prediction", 0), i)
        .Split(user_op::OpArg("label", 0), i)
        .Split(user_op::OpArg("prob", 0), i)
        .Split(user_op::OpArg("out", 0), i)
        .Build();
  }
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> SoftmaxCrossEntropyOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& prediction_desc = ctx->InputTensorDesc("prediction", 0);
  const user_op::TensorDesc& label_desc = ctx->InputTensorDesc("label", 0);
  CHECK_EQ_OR_RETURN(prediction_desc.is_dynamic(), label_desc.is_dynamic())
      << Error::RuntimeError()
      << "prediction and label are expected to have the same dynamic property, but found "
      << prediction_desc.is_dynamic() << " and " << label_desc.is_dynamic();
  CHECK_GE_OR_RETURN(prediction_desc.shape().NumAxes(), 2)
      << Error::RuntimeError()
      << "The dimension of prediction must be greater than or equal to 2, but found "
      << prediction_desc.shape().NumAxes();
  CHECK_EQ_OR_RETURN(label_desc.shape(), prediction_desc.shape())
      << Error::RuntimeError() << "The size of label " << label_desc.shape()
      << " must match the size of prediction " << prediction_desc.shape();
  const int64_t num_out_axes = prediction_desc.shape().NumAxes() - 1;
  DimVector out_dim_vector;
  FOR_RANGE(int64_t, i, 0, num_out_axes) {
    out_dim_vector.emplace_back(prediction_desc.shape().At(i));
  }
  ctx->SetOutputShape("prob", 0, ctx->InputShape("prediction", 0));
  ctx->SetOutputIsDynamic("prob", 0, ctx->InputIsDynamic("prediction", 0));
  user_op::TensorDesc* out_desc = ctx->MutOutputTensorDesc("out", 0);
  out_desc->set_is_dynamic(prediction_desc.is_dynamic());
  out_desc->set_shape(Shape(out_dim_vector));
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> SoftmaxCrossEntropyOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> SoftmaxCrossEntropyOp::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& prediction_desc = ctx->InputTensorDesc("prediction", 0);
  const user_op::TensorDesc& label_desc = ctx->InputTensorDesc("label", 0);
  CHECK_EQ_OR_RETURN(label_desc.data_type(), prediction_desc.data_type())
      << Error::TypeError()
      << "label and prediction are expected to have the same dtype, but found "
      << DataType_Name(label_desc.data_type()) << " and "
      << DataType_Name(prediction_desc.data_type());
  ctx->SetOutputDType("prob", 0, ctx->InputDType("prediction", 0));
  user_op::TensorDesc* out_desc = ctx->MutOutputTensorDesc("out", 0);
  out_desc->set_data_type(prediction_desc.data_type());
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> SoftmaxCrossEntropyOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper&) {
  user_op::InputArgModifier* cond_arg_modifier = GetInputArgModifierFn("label", 0);
  cond_arg_modifier->set_requires_grad(false);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> SoftmaxCrossEntropyGradOp::GetSbp(user_op::SbpContext* ctx) {
  const auto num_dy_axes = ctx->LogicalTensorDesc4InputArgNameAndIndex("dy", 0).shape().NumAxes();
  FOR_RANGE(int64_t, i, 0, num_dy_axes) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("dy", 0), i)
        .Split(user_op::OpArg("label", 0), i)
        .Split(user_op::OpArg("prob", 0), i)
        .Split(user_op::OpArg("prediction_diff", 0), i)
        .Build();
  }
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> SoftmaxCrossEntropyGradOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  const user_op::TensorDesc& prob_desc = ctx->InputTensorDesc("prob", 0);
  const user_op::TensorDesc& label_desc = ctx->InputTensorDesc("label", 0);
  const user_op::TensorDesc& dy_desc = ctx->InputTensorDesc("dy", 0);
  CHECK_EQ_OR_RETURN(prob_desc.is_dynamic(), label_desc.is_dynamic())
      << Error::RuntimeError()
      << "prob and label are expected to have the same dynamic property, but found "
      << prob_desc.is_dynamic() << " and " << label_desc.is_dynamic();
  CHECK_GE_OR_RETURN(prob_desc.shape().NumAxes(), 2)
      << Error::RuntimeError()
      << "The dimension of prob must be greater than or equal to 2, but found "
      << prob_desc.shape().NumAxes();
  CHECK_EQ_OR_RETURN(dy_desc.shape().NumAxes(), prob_desc.shape().NumAxes() - 1)
      << Error::RuntimeError()
      << "The dimension of dy is expected to be less than that of prob by 1, but found "
      << dy_desc.shape().NumAxes() << " and " << prob_desc.shape().NumAxes() - 1;
  FOR_RANGE(int64_t, i, 0, dy_desc.shape().NumAxes()) {
    CHECK_EQ_OR_RETURN(dy_desc.shape().At(i), label_desc.shape().At(i))
        << Error::RuntimeError() << "The size of dy (" << dy_desc.shape().At(i)
        << ") must match the size of label (" << label_desc.shape().At(i) << ") at dimension " << i;
  }
  CHECK_EQ_OR_RETURN(label_desc.shape(), prob_desc.shape())
      << Error::RuntimeError() << "The size of label " << label_desc.shape()
      << " must match the size of prob " << prob_desc.shape();
  ctx->SetOutputShape("prediction_diff", 0, ctx->InputShape("prob", 0));
  ctx->SetOutputIsDynamic("prediction_diff", 0, ctx->InputIsDynamic("prob", 0));
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> SoftmaxCrossEntropyGradOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> SoftmaxCrossEntropyGradOp::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& prob_desc = ctx->InputTensorDesc("prob", 0);
  const user_op::TensorDesc& label_desc = ctx->InputTensorDesc("label", 0);
  const user_op::TensorDesc& dy_desc = ctx->InputTensorDesc("dy", 0);
  CHECK_EQ_OR_RETURN(label_desc.data_type(), prob_desc.data_type())
      << Error::TypeError() << "label and prob are expected to have the same dtype, but found "
      << DataType_Name(label_desc.data_type()) << " and " << DataType_Name(prob_desc.data_type());
  CHECK_EQ_OR_RETURN(dy_desc.data_type(), prob_desc.data_type())
      << Error::TypeError() << "dy and prob are expected to have the same dtype, but found "
      << DataType_Name(dy_desc.data_type()) << " and " << DataType_Name(prob_desc.data_type());
  ctx->SetOutputDType("prediction_diff", 0, ctx->InputDType("prob", 0));
  return Maybe<void>::Ok();
}

}  // namespace oneflow
