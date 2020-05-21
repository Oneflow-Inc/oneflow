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

Maybe<void> ForeignWatchOp::InferSbpSignature(
    SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
    const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
    std::function<Maybe<const SbpInferHint*>(const std::string&)> SbpInferHint4Ibn,
    const ParallelDesc& parallel_desc) const {
  CHECK_EQ_OR_RETURN(JUST(SbpInferHint4Ibn("in"))->parallel_desc().parallel_num(), 1);
  CHECK_OR_RETURN(JUST(SbpInferHint4Ibn("in"))->parallel_desc() == parallel_desc);
  (*sbp_signature->mutable_bn_in_op2sbp_parallel())["in"].mutable_split_parallel()->set_axis(0);
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kForeignWatchConf, ForeignWatchOp);

}  // namespace oneflow
