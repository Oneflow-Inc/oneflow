#include "oneflow/core/framework/framework.h"
#include "oneflow/core/operator/reshape_op_util.h"

namespace oneflow {

REGISTER_USER_OP("reshape_like")
    .Input("in")
    .Input("like")
    .Output("out")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape* in_shape = ctx->Shape4ArgNameAndIndex("in", 0);
      const Shape* like_shape = ctx->Shape4ArgNameAndIndex("like", 0);
      CHECK_EQ_OR_RETURN(in_shape->elem_cnt(), like_shape->elem_cnt());
      *ctx->Shape4ArgNameAndIndex("out", 0) = *like_shape;
      *ctx->Dtype4ArgNameAndIndex("out", 0) = *ctx->Dtype4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    })
    .SetInputArgModifyFn([](user_op::GetInputArgModifier GetInputArgModifierFn) {
      user_op::InputArgModifier* like_arg_modifier = GetInputArgModifierFn("like", 0);
      CHECK(like_arg_modifier != nullptr);
      like_arg_modifier->set_use_header_only(true);
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      *ctx->BatchAxis4ArgNameAndIndex("out", 0) = *ctx->BatchAxis4ArgNameAndIndex("like", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const auto& in_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0).shape();
      const auto& like_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("like", 0).shape();
      SbpSignatureBuilder().PartialSum("like").Broadcast("in").Broadcast("out").Build(
          ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
      SbpSignatureBuilder().Broadcast("like").PartialSum("in").PartialSum("out").Build(
          ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
      return ReshapeOpUtil::GetReshapeSbpSignatures(
          in_shape, like_shape, StdVec2PbRpf<std::string>({"in"}),
          StdVec2PbRpf<std::string>({"like", "out"}), ctx->parallel_num(), ctx->sbp_sig_list());
    });

}  // namespace oneflow
