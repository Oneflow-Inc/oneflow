#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace {

int64_t ShiftNegativeAxisIfNeed(const Shape& shape, int64_t axis) {
  const int64_t shifted = axis < 0 ? axis + shape.NumAxes() : axis;
  CHECK_GE(shifted, 0);
  CHECK_LT(shifted, shape.NumAxes());
  return shifted;
}

}  // namespace

REGISTER_USER_OP("broadcast_div_grad")
    .Input("b")
    .Input("y")
    .Input("dy")
    .Output("db")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* b = ctx->TensorDesc4ArgNameAndIndex("b", 0);
      user_op::TensorDesc* db = ctx->TensorDesc4ArgNameAndIndex("db", 0);
      *db = *b;
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      *ctx->BatchAxis4ArgNameAndIndex("db", 0) = *ctx->BatchAxis4ArgNameAndIndex("b", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      TODO();
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
