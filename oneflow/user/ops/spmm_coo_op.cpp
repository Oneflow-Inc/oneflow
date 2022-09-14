#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

/* static */ Maybe<void> SpmmCooOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  int64_t a_row = ctx->Attr<int64_t>("a_rows");
  const user_op::TensorDesc& b = ctx->InputTensorDesc("b", 0);
  int64_t b_col = b.shape().At(1);

  *ctx->MutOutputShape("out", 0) = {a_row, b_col};

  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> SpmmCooOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> SpmmCooOp::GetSbp(user_op::SbpContext* ctx) {
    // ctx->NewBuilder()
    //   .Split(user_op::OpArg("a_cooRowInd", 0), 0)
    //   .Split(user_op::OpArg("a_cooColInd", 0), 0)
    //   .Split(user_op::OpArg("a_cooValues", 0), 0)
    //   .Broadcast(user_op::OpArg("b", 0))
    //   .Split(user_op::OpArg("out", 0), 0)
    //   .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> SpmmCooOp::InferDataType(user_op::InferContext* ctx) {
 
  *ctx->MutOutputDType("out", 0) = ctx->InputDType("b", 0);

  return Maybe<void>::Ok();
}

}  // namespace oneflow
