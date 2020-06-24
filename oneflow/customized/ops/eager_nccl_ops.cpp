#include "oneflow/core/framework/framework.h"

namespace oneflow {

REGISTER_USER_OP("eager_nccl_all_reduce")
    .Input("in")
    .Output("out")
    .Attr("device_set_machine_ids", UserOpAttrType::kAtListInt64)
    .Attr("device_set_device_ids", UserOpAttrType::kAtListInt64)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->Shape4ArgNameAndIndex("out", 0) = *ctx->Shape4ArgNameAndIndex("in", 0);
      *ctx->Dtype4ArgNameAndIndex("out", 0) = *ctx->Dtype4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder()
          .PartialSum(user_op::OpArg("in", 0))
          .Broadcast(user_op::OpArg("out", 0))
          .Build();
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
