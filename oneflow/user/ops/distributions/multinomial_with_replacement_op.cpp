#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

/* static */ Maybe<void> MultinomialWithReplacementOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  int32_t num_samples = ctx->Attr<int32_t>("num_samples");
  const Shape& x_shape = ctx->InputShape("x", 0);
  if (x_shape.NumAxes() == 1) {
    ctx->SetOutputShape("out", 0, Shape({num_samples}));
  } else {
    ctx->SetOutputShape("out", 0, Shape({x_shape.At(0), num_samples}));
  }
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> MultinomialWithReplacementOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> MultinomialWithReplacementOp::GetSbp(user_op::SbpContext* ctx) {
  const Shape& x_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0).shape();
  if (x_shape.NumAxes() == 2) {
    ctx->NewBuilder()
            .Split(ctx->inputs(), 0)
            .Split(ctx->outputs(), 0)
            .Build();
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> MultinomialWithReplacementOp::InferDataType(user_op::InferContext* ctx) {
  DataType in_types = ctx->InputDType("x", 0);
  ctx->SetOutputDType("out", 0, DataType::kInt64);
  return Maybe<void>::Ok();
}

}  // namespace oneflow
