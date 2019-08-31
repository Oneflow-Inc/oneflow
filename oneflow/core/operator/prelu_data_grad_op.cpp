#include "oneflow/core/operator/prelu_data_grad_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void PReluDataGradOp::InitFromOpConf() {
  CHECK(op_conf().has_prelu_data_grad_conf());
  EnrollInputBn("dy", false);
  EnrollInputBn("alpha", false);
  EnrollInputBn("x", false);
  EnrollOutputBn("dx", false);
}

const PbMessage& PReluDataGradOp::GetCustomizedConf() const {
  return op_conf().prelu_data_grad_conf();
}

Maybe<void> PReluDataGradOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("dx") = *GetBlobDesc4BnInOp("x");
  return Maybe<void>::Ok();
}

Maybe<void> PReluDataGradOp::InferHasBatchDim(
    std::function<bool*(const std::string&)> HasBatchDim4BnInOp) const {
  CHECK(*HasBatchDim4BnInOp("dy"));
  CHECK(*HasBatchDim4BnInOp("x"));
  CHECK(*HasBatchDim4BnInOp("alpha") == false);
  *HasBatchDim4BnInOp("dx") = true;
  return Maybe<void>::Ok();
}
void PReluDataGradOp::GetSbpSignatures(
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split("dy", 0)
      .Broadcast("alpha")
      .Split("x", 0)
      .Split("dx", 0)
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
}

REGISTER_OP(OperatorConf::kPreluDataGradConf, PReluDataGradOp);

}  // namespace oneflow
