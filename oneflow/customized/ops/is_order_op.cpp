#include "oneflow/core/framework/framework.h"

namespace oneflow {

REGISTER_USER_OP("is_order")
    .Input("in")
    .Output("out")
    .Attr("order_type", UserOpAttrType::kAtString)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->Shape4ArgNameAndIndex("out", 0) = Shape({1});
      *ctx->Dtype4ArgNameAndIndex("out", 0) = kInt8;
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      CHECK_OR_RETURN(!ctx->BatchAxis4ArgNameAndIndex("in", 0)->has_value())
          << "The batch axis value is " << ctx->BatchAxis4ArgNameAndIndex("in", 0)->value();
      ctx->BatchAxis4ArgNameAndIndex("out", 0)->clear_value();
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      return Maybe<void>::Ok();  // with default (B，…)->(B，…)
    })
    .SetCheckAttrFn([](const user_op::UserOpDefWrapper& op_def,
                       const user_op::UserOpConfWrapper& op_conf) -> Maybe<void> {
      const std::string& order_type = op_conf.attr<std::string>("order_type");
      CHECK_OR_RETURN(order_type == "NON_DECREASING" || order_type == "STRICTLY_INCREASING");
      return Maybe<void>::Ok();
    });

}  // namespace oneflow