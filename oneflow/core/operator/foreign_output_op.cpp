#include "oneflow/core/operator/foreign_output_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void ForeignOutputOp::InitFromOpConf() {
  CHECK(op_conf().has_foreign_output_conf());
  EnrollInputBn("in");
}

Maybe<void> ForeignOutputOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  CHECK_EQ_OR_RETURN(parallel_ctx->parallel_num(), 1);
  return Maybe<void>::Ok();
}

const PbMessage& ForeignOutputOp::GetCustomizedConf() const {
  return op_conf().foreign_output_conf();
}

Maybe<void> ForeignOutputOp::InferBatchAxis(
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  return Maybe<void>::Ok();
}

void ForeignOutputOp::GetSbpSignatures(
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {}

REGISTER_OP(OperatorConf::kForeignOutputConf, ForeignOutputOp);

}  // namespace oneflow
