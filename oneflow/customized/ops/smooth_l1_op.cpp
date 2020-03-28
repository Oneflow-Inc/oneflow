#include "oneflow/core/framework/framework.h"

namespace oneflow {

REGISTER_USER_OP("smooth_l1")
    .Input("x")
    .Input("label")
    .Output("y")
    .Attr("beta", UserOpAttrType::kAtFloat)
    .Attr("scale", UserOpAttrType::kAtFloat)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->Shape4ArgNameAndIndex("y", 0) = Shape({ctx->Shape4ArgNameAndIndex("x", 0)->At(0)});
      CHECK_EQ_OR_RETURN(*ctx->Shape4ArgNameAndIndex("x", 0),
                         *ctx->Shape4ArgNameAndIndex("label", 0));
      CHECK_EQ_OR_RETURN(*ctx->Dtype4ArgNameAndIndex("x", 0),
                         *ctx->Dtype4ArgNameAndIndex("label", 0));
      CHECK_GE_OR_RETURN(ctx->GetAttr<float>("beta"), 0);
      *ctx->Shape4ArgNameAndIndex("y", 0) = *ctx->Shape4ArgNameAndIndex("x", 0);
      *ctx->Dtype4ArgNameAndIndex("y", 0) = *ctx->Dtype4ArgNameAndIndex("x", 0);
      return Maybe<void>::Ok();
    });
}  // namespace oneflow
