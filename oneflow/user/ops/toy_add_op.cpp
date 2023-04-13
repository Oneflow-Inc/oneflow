#include "oneflow/core/common/maybe.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

/* static */ Maybe<void> ToyAdd::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const auto& x = ctx->InputTensorDesc("x", 0);
  const auto& y = ctx->InputTensorDesc("y", 0);
  auto out = ctx->MutOutputTensorDesc("output", 0);
  CHECK_EQ_OR_RETURN(x.shape(), y.shape())<< Error::RuntimeError()<<"x and y must be same tensor size";
  out->set_shape(x.shape());
  out->set_is_dynamic(x.is_dynamic());
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> ToyAdd::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> ToyAdd::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& x_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
  FOR_RANGE(int64_t, i, 0, x_tensor.shape().NumAxes()) {
    ctx->NewBuilder().Split(user_op::OpArg("x", 0), i).Split(user_op::OpArg("output", 0), i).Build();
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> ToyAdd::InferDataType(user_op::InferContext* ctx) {
  const auto& x = ctx->InputTensorDesc("x", 0);
  const auto& y = ctx->InputTensorDesc("y", 0);
  auto out = ctx->MutOutputTensorDesc("output", 0);
  CHECK_EQ_OR_RETURN(x.shape(), y.shape())<< Error::RuntimeError()<<"x and y must be same tensor size";
  out->set_data_type(x.data_type());
  return Maybe<void>::Ok();
}

}  // namespace oneflow