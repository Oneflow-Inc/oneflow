#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

Maybe<void> ThrowErrorOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return user_op::TensorDescInferFnUtil::Unchanged(ctx);
}
Maybe<void> ThrowErrorOp::GetSbp(user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast(ctx);
}
Maybe<void> ThrowErrorOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->MutOutputDType("y", 0) = ctx->InputDType("x", 0);
  return Maybe<void>::Ok();
}

}  // namespace oneflow
