#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

/* static */ Maybe<void> MOEDispatchOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& in_shape = ctx->InputShape("in", 0);
  const int32_t hidden_size = in_shape.at(1);

  const int32_t num_experts = ctx->Attr<int32_t>("num_experts");
  // FIXME(wangxi): capacity should be a tensor
  const int32_t capacity = ctx->Attr<int32_t>("capacity");
  CHECK_GT_OR_RETURN(num_experts, 0);
  CHECK_GT_OR_RETURN(capacity, 0);

  ctx->SetOutputShape("out", 0, Shape({num_expert, capacity, hidden_size}));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> MOEDispatchOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> MOEDispatchOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("y", 0) = ctx->InputDType("x", 0);
  return Maybe<void>::Ok();
}

}  // namespace oneflow