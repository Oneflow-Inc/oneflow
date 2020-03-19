#include "oneflow/core/framework/framework.h"

namespace oneflow {

REGISTER_USER_OP("identity")
    .Input("in")
    .Output("out")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->Shape4ArgNameAndIndex("out", 0) = *ctx->Shape4ArgNameAndIndex("in", 0);
      *ctx->Dtype4ArgNameAndIndex("out", 0) = *ctx->Dtype4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      SbpSignatureBuilder()
          .Split("in", 0)
          .Split("out", 0)
          .MakeSplitSignatureListBuilder(
              ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0).shape().NumAxes())
          .Build(ctx->sbp_sig_list());
      SbpSignatureBuilder().PartialSum("in").PartialSum("out").Build(
          ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
