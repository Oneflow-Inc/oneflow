#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

/********** MOEDispatch Op ***********/
/* static */ Maybe<void> MOEDispatchOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& in_shape = ctx->InputShape("in", 0);
  const int32_t hidden_size = in_shape.at(1);

  const int32_t num_experts = ctx->Attr<int32_t>("num_experts");
  // FIXME(wangxi): capacity should be a tensor
  const int32_t capacity = ctx->Attr<int32_t>("capacity");
  CHECK_GT_OR_RETURN(num_experts, 0);
  CHECK_GT_OR_RETURN(capacity, 0);

  ctx->SetOutputShape("out", 0, Shape({num_experts, capacity, hidden_size}));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> MOEDispatchOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> MOEDispatchOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, ctx->InputDType("in", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> MOEDispatchOp::GetSbp(user_op::SbpContext* ctx) { return Maybe<void>::Ok(); }

/********** MOECombine Op ***********/
/* static */ Maybe<void> MOECombineOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& in_shape = ctx->InputShape("in", 0);
  const int32_t hidden_size = in_shape.at(2);

  const Shape& indices_shape = ctx->InputShape("indices", 0);
  const int32_t samples = indices_shape.at(indices_shape.NumAxes() - 1);

  ctx->SetOutputShape("out", 0, Shape({samples, hidden_size}));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> MOECombineOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> MOECombineOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, ctx->InputDType("in", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> MOECombineOp::GetSbp(user_op::SbpContext* ctx) { return Maybe<void>::Ok(); }

/********** MOEGateGrad Op ***********/
/* static */ Maybe<void> MOEGateGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  ctx->SetOutputShape("out", 0, ctx->InputShape("indices", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> MOEGateGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> MOEGateGradOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, ctx->InputDType("in", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> MOEGateGradOp::GetSbp(user_op::SbpContext* ctx) { return Maybe<void>::Ok(); }


}  // namespace oneflow
