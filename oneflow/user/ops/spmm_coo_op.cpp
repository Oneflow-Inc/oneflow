#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

/* static */ Maybe<void> SpmmCooOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  int64_t a_row = ctx->Attr<int64_t>("a_num_rows");
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
    //   .Split(user_op::OpArg("a_coo_row", 0), 0)
    //   .Split(user_op::OpArg("a_coo_col", 0), 0)
    //   .Split(user_op::OpArg("a_coo_val", 0), 0)
    //   .Broadcast(user_op::OpArg("b", 0))
    //   .Split(user_op::OpArg("out", 0), 0)
    //   .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> SpmmCooOp::InferDataType(user_op::InferContext* ctx) {
 
  *ctx->MutOutputDType("out", 0) = ctx->InputDType("b", 0);

  return Maybe<void>::Ok();
}

/* static */ Maybe<void> SpmmCooGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  int64_t a_row = ctx->Attr<int64_t>("a_num_rows");
  const user_op::TensorDesc& dout = ctx->InputTensorDesc("dout", 0);
  int64_t dout_col = dout.shape().At(1);
  
  *ctx->MutOutputShape("db", 0) = {a_row, dout_col};
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> SpmmCooGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> SpmmCooGradOp::GetSbp(user_op::SbpContext* ctx) {

  return Maybe<void>::Ok();
}

/* static */ Maybe<void> SpmmCooGradOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->MutOutputDType("db", 0) = ctx->InputDType("dout", 0);
  return Maybe<void>::Ok();
}

}  // namespace oneflow
