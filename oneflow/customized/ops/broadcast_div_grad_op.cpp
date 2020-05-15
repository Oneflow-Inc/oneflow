#include "oneflow/core/framework/framework.h"

namespace oneflow {

REGISTER_USER_OP("broadcast_div_grad")
    .Input("y")
    .Input("z")
    .Input("dz")
    .Output("dy")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->TensorDesc4ArgNameAndIndex("dy", 0) = *ctx->TensorDesc4ArgNameAndIndex("y", 0);
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn(user_op::BatchAxisInferFnUtil::NaiveInferBatchAxis)
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder()
          .Broadcast(user_op::OpArg("y", 0))
          .PartialSum(user_op::OpArg("z", 0))
          .Broadcast(user_op::OpArg("dz", 0))
          .Broadcast(user_op::OpArg("dy", 0))
          .Build();
      ctx->NewBuilder()
          .Broadcast(user_op::OpArg("y", 0))
          .Broadcast(user_op::OpArg("z", 0))
          .PartialSum(user_op::OpArg("dz", 0))
          .Broadcast(user_op::OpArg("dy", 0))
          .Build();
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
