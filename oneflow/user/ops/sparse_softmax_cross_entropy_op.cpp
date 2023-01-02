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

namespace {

Maybe<void> InferTensorDescFn(user_op::InferContext* ctx) {
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
  const int64_t num_out_axes = prediction_desc.shape().NumAxes() - 1;
  CHECK_EQ_OR_RETURN(label_desc.shape().NumAxes(), num_out_axes)
      << Error::RuntimeError()
      << "The dimension of label is expected to be less than that of prediction by 1, but found "
      << label_desc.shape().NumAxes() << " and " << num_out_axes;
  FOR_RANGE(int64_t, i, 0, num_out_axes) {
    CHECK_EQ_OR_RETURN(prediction_desc.shape().At(i), label_desc.shape().At(i))
        << Error::RuntimeError() << "The size of prediction (" << prediction_desc.shape().At(i)
        << ") must match the size of label (" << label_desc.shape().At(i) << ") at dimension " << i;
  }
  ctx->SetOutputIsDynamic("prob", 0, prediction_desc.is_dynamic());
  // 'prob' is just for compute prediction's grad, prob's grad will be ignored
  ctx->SetOutputShape("prob", 0, prediction_desc.shape());
  user_op::TensorDesc* out_desc = ctx->MutOutputTensorDesc("out", 0);
  out_desc->set_is_dynamic(prediction_desc.is_dynamic());
  out_desc->set_shape(label_desc.shape());
  return Maybe<void>::Ok();
}

Maybe<void> InferGradTensorDescFn(user_op::InferContext* ctx) {
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
  const int64_t num_out_axes = prob_desc.shape().NumAxes() - 1;
  CHECK_EQ_OR_RETURN(label_desc.shape().NumAxes(), num_out_axes)
      << Error::RuntimeError()
      << "The dimension of label is expected to be less than that of prediction by 1, but found "
      << label_desc.shape().NumAxes() << " and " << num_out_axes;
  FOR_RANGE(int64_t, i, 0, num_out_axes) {
    CHECK_EQ_OR_RETURN(prob_desc.shape().At(i), label_desc.shape().At(i))
        << Error::RuntimeError() << "The size of prob (" << prob_desc.shape().At(i)
        << ") must match the size of label (" << label_desc.shape().At(i) << ") at dimension " << i;
  }
  CHECK_EQ_OR_RETURN(dy_desc.shape(), label_desc.shape())
      << Error::RuntimeError() << "The size of dy " << dy_desc.shape()
      << " must match the size of label " << label_desc.shape();
  ctx->SetOutputShape("prediction_diff", 0, prob_desc.shape());
  ctx->SetOutputIsDynamic("prediction_diff", 0, prob_desc.is_dynamic());
  return Maybe<void>::Ok();
}

Maybe<void> InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& label_desc = ctx->InputTensorDesc("label", 0);
  CHECK_OR_RETURN(IsIndexDataType(label_desc.data_type()))
      << Error::TypeError() << "The dtype of label must be integer, but found "
      << DataType_Name(label_desc.data_type());
  ctx->SetOutputDType("prob", 0, ctx->InputDType("prediction", 0));
  ctx->SetOutputDType("out", 0, ctx->InputDType("prediction", 0));
  return Maybe<void>::Ok();
}

Maybe<void> InferDataTypeGrad(user_op::InferContext* ctx) {
  const user_op::TensorDesc& prob_desc = ctx->InputTensorDesc("prob", 0);
  const user_op::TensorDesc& label_desc = ctx->InputTensorDesc("label", 0);
  CHECK_OR_RETURN(IsIndexDataType(label_desc.data_type()))
      << Error::TypeError() << "The dtype of label must be integer, but found "
      << DataType_Name(label_desc.data_type());
  const user_op::TensorDesc& dy_desc = ctx->InputTensorDesc("dy", 0);
  CHECK_EQ_OR_RETURN(dy_desc.data_type(), prob_desc.data_type())
      << Error::TypeError() << "dy and prob are expected to have the same dtype, but found "
      << DataType_Name(dy_desc.data_type()) << " and " << DataType_Name(prob_desc.data_type());
  ctx->SetOutputDType("prediction_diff", 0, prob_desc.data_type());
  return Maybe<void>::Ok();
}

Maybe<void> AddSignature(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Split(user_op::OpArg("prediction", 0), 0)
      .Split(user_op::OpArg("label", 0), 0)
      .Split(user_op::OpArg("prob", 0), 0)
      .Split(user_op::OpArg("out", 0), 0)
      .Build();
  return Maybe<void>::Ok();
}

Maybe<void> AddMsSignature(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& prediction =
      ctx->LogicalTensorDesc4InputArgNameAndIndex("prediction", 0);
  ctx->NewBuilder()
      .Split(user_op::OpArg("prediction", 0), 0)
      .Split(user_op::OpArg("prob", 0), 0)
      .Split(user_op::OpArg("label", 0), 0)
      .Split(user_op::OpArg("out", 0), 0)
      .Build();
  ctx->NewBuilder()
      .Split(user_op::OpArg("prediction", 0), prediction.shape().NumAxes() - 1)
      .Split(user_op::OpArg("prob", 0), prediction.shape().NumAxes() - 1)
      .Broadcast(user_op::OpArg("label", 0))
      .PartialSum(user_op::OpArg("out", 0))
      .Build();
  return Maybe<void>::Ok();
}

Maybe<void> AddGradSignature(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Split(user_op::OpArg("dy", 0), 0)
      .Split(user_op::OpArg("label", 0), 0)
      .Split(user_op::OpArg("prob", 0), 0)
      .Split(user_op::OpArg("prediction_diff", 0), 0)
      .Build();
  return Maybe<void>::Ok();
}

Maybe<void> AddGradMsSignature(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& prob = ctx->LogicalTensorDesc4InputArgNameAndIndex("prob", 0);
  ctx->NewBuilder()
      .Split(user_op::OpArg("prob", 0), 0)
      .Split(user_op::OpArg("label", 0), 0)
      .Split(user_op::OpArg("dy", 0), 0)
      .Split(user_op::OpArg("prediction_diff", 0), 0)
      .Build();
  ctx->NewBuilder()
      .Split(user_op::OpArg("prob", 0), prob.shape().NumAxes() - 1)
      .Broadcast(user_op::OpArg("label", 0))
      .Broadcast(user_op::OpArg("dy", 0))
      .Split(user_op::OpArg("prediction_diff", 0), prob.shape().NumAxes() - 1)
      .Build();
  return Maybe<void>::Ok();
}

template<Maybe<void> (*GetSbpSignature)(user_op::SbpContext*)>
Maybe<void> GetSbpFn(user_op::SbpContext* ctx) {
  JUST(GetSbpSignature(ctx));
  return Maybe<void>::Ok();
}

}  // namespace

#define IMPLEMENT_SPAESE_SOFTMAX_CROSS_ENTROPY_OP_FUNCS(op_name, sbp_sig)                       \
  /*static*/ Maybe<void> op_name##Op::GetSbp(user_op::SbpContext* ctx) { return sbp_sig(ctx); } \
  /*static*/ Maybe<void> op_name##Op::InferLogicalTensorDesc(user_op::InferContext* ctx) {      \
    return InferTensorDescFn(ctx);                                                              \
  }                                                                                             \
  /*static*/ Maybe<void> op_name##Op::InferPhysicalTensorDesc(user_op::InferContext* ctx) {     \
    return InferLogicalTensorDesc(ctx);                                                         \
  }                                                                                             \
  /*static*/ Maybe<void> op_name##Op::InferDataType(user_op::InferContext* ctx) {               \
    return oneflow::InferDataType(ctx);                                                         \
  }                                                                                             \
  /*static*/ Maybe<void> op_name##Op::ModifyInputArg(                                           \
      const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper&) {    \
    user_op::InputArgModifier* label_modifier = GetInputArgModifierFn("label", 0);              \
    CHECK_OR_RETURN(label_modifier != nullptr); /* NOLINT(maybe-need-error-msg) */              \
    label_modifier->set_requires_grad(false);                                                   \
    return Maybe<void>::Ok();                                                                   \
  }

IMPLEMENT_SPAESE_SOFTMAX_CROSS_ENTROPY_OP_FUNCS(SparseSoftmaxCrossEntropy, AddSignature);
IMPLEMENT_SPAESE_SOFTMAX_CROSS_ENTROPY_OP_FUNCS(SparseSoftmaxCrossEntropyMs, AddMsSignature);
#undef IMPLEMENT_SPAESE_SOFTMAX_CROSS_ENTROPY_OP_FUNCS

#define IMPLEMENT_SPAESE_SOFTMAX_CROSS_ENTROPY_GRAD_OP_FUNCS(op_name, sbp_sig)                  \
  /*static*/ Maybe<void> op_name##GradOp::GetSbp(user_op::SbpContext* ctx) {                    \
    return sbp_sig(ctx);                                                                        \
  }                                                                                             \
  /*static*/ Maybe<void> op_name##GradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {  \
    return InferGradTensorDescFn(ctx);                                                          \
  }                                                                                             \
  /*static*/ Maybe<void> op_name##GradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) { \
    return InferLogicalTensorDesc(ctx);                                                         \
  }                                                                                             \
  /*static*/ Maybe<void> op_name##GradOp::InferDataType(user_op::InferContext* ctx) {           \
    return InferDataTypeGrad(ctx);                                                              \
  }

IMPLEMENT_SPAESE_SOFTMAX_CROSS_ENTROPY_GRAD_OP_FUNCS(SparseSoftmaxCrossEntropy, AddGradSignature);
IMPLEMENT_SPAESE_SOFTMAX_CROSS_ENTROPY_GRAD_OP_FUNCS(SparseSoftmaxCrossEntropyMs,
                                                     AddGradMsSignature);
#undef IMPLEMENT_SPAESE_SOFTMAX_CROSS_ENTROPY_GRAD_OP_FUNCS

}  // namespace oneflow
