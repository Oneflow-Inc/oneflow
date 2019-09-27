#include "oneflow/core/operator/foreign_watch_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void ForeignWatchOp::InitFromOpConf() {
  CHECK(op_conf().has_foreign_watch_conf());
  EnrollInputBn("in");
}

Maybe<void> ForeignWatchOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  CHECK_EQ_OR_RETURN(parallel_ctx->parallel_num(), 1);
  return Maybe<void>::Ok();
}

const PbMessage& ForeignWatchOp::GetCustomizedConf() const {
  return op_conf().foreign_watch_conf();
}

Maybe<void> ForeignWatchOp::InferBatchAxis(
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  return Maybe<void>::Ok();
}

Maybe<void> ForeignWatchOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kForeignWatchConf, ForeignWatchOp);

}  // namespace oneflow
