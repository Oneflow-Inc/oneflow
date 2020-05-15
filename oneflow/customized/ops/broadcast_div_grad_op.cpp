#include "oneflow/core/framework/framework.h"

namespace oneflow {

REGISTER_USER_OP("broadcast_div_grad")
    .Input("y")
    .Input("z")
    .Input("dz")
    .Output("dy")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* b = ctx->TensorDesc4ArgNameAndIndex("y", 0);
      user_op::TensorDesc* db = ctx->TensorDesc4ArgNameAndIndex("dy", 0);
      *db = *b;
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      *ctx->BatchAxis4ArgNameAndIndex("dy", 0) = *ctx->BatchAxis4ArgNameAndIndex("y", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      TODO();
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
