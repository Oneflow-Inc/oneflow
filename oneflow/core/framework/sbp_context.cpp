#include "oneflow/core/framework/sbp_context.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

namespace user_op {

Maybe<void> GetSbpFnUtil::MirrorSplitAtDim0(SbpContext* ctx) {
  SbpSignatureBuilder()
      .Split(ctx->inputs(), 0)
      .Split(ctx->outputs(), 0)
      .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

}  // namespace user_op

}  // namespace oneflow
