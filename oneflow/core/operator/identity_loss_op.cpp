#include "oneflow/core/operator/identity_loss_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

const PbMessage& IdentityLossOp::GetCustomizedConf() const {
  return op_conf().identity_loss_conf();
}

LossKernelConf* IdentityLossOp::GetMutLossKernelConf(KernelConf* kernel_conf) const {
  return kernel_conf->mutable_identity_loss_conf()->mutable_loss_conf();
}

Maybe<void> IdentityLossOp::VirtualInferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* prediction = GetBlobDesc4BnInOp("prediction");
  *GetBlobDesc4BnInOp("loss") = *prediction;
  return Maybe<void>::Ok();
}

void IdentityLossOp::GetSbpSignatures(SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split(input_bns(), 0)
      .Split(output_bns(), 0)
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  SbpSignatureBuilder()
      .PartialSum(input_bns())
      .PartialSum(output_bns())
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
}

REGISTER_OP(OperatorConf::kIdentityLossConf, IdentityLossOp);

}  // namespace oneflow
