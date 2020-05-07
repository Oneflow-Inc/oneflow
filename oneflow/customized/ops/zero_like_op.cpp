#include "oneflow/core/framework/framework.h"

namespace oneflow {

REGISTER_USER_OP("zero_like")
    .Input("like")
    .Output("out")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->Shape4ArgNameAndIndex("out", 0) = *ctx->Shape4ArgNameAndIndex("like", 0);
      *ctx->Dtype4ArgNameAndIndex("out", 0) = *ctx->Dtype4ArgNameAndIndex("like", 0);
      return Maybe<void>::Ok();
    })
    .SetInputArgModifyFn([](user_op::GetInputArgModifier GetInputArgModifierFn) {
      user_op::InputArgModifier* like_arg_modifier = GetInputArgModifierFn("like", 0);
      CHECK(like_arg_modifier != nullptr);
      like_arg_modifier->set_use_header_only(true);
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      SbpSignatureBuilder()
          .Split("like", 0, 0)
          .Split("out", 0, 0)
          .MakeSplitSignatureListBuilder(
              ctx->LogicalTensorDesc4InputArgNameAndIndex("like", 0).shape().NumAxes())
          .Build(ctx->sbp_sig_list());
      SbpSignatureBuilder().PartialSum("like", 0).Broadcast("out", 0).Build(
          ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
