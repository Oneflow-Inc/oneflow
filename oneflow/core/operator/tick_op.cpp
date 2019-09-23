#include "oneflow/core/operator/tick_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void TickOp::InitFromOpConf() {
  EnrollRepeatedInputBn("tick", false);
  EnrollOutputBn("out", false);
}

Maybe<void> TickOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                   const ParallelContext* parallel_ctx) const {
  GetBlobDesc4BnInOp("out")->mut_shape() = Shape({1});
  return Maybe<void>::Ok();
}

Maybe<void> TickOp::InferBatchAxis(
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  BatchAxis4BnInOp("out")->clear_value();
  return Maybe<void>::Ok();
}

Maybe<void> TickOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  return Maybe<void>::Ok();
}

REGISTER_OP_SAME_OUTPUT_BLOB_REGST_NUM(OperatorConf::kTickConf, 2);
REGISTER_OP(OperatorConf::kTickConf, TickOp);
REGISTER_TICK_TOCK_OP(OperatorConf::kTickConf);

}  // namespace oneflow
