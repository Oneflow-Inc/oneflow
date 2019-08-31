#include "oneflow/core/operator/sigmoid_grad_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void SigmoidGradOp::InitFromOpConf() {
  CHECK(op_conf().has_sigmoid_grad_conf());
  EnrollInputBn("y");
  EnrollInputBn("dy");
  EnrollOutputBn("dx")->set_mutable_inplace_ibn("dy");
}

const PbMessage& SigmoidGradOp::GetCustomizedConf() const { return op_conf().sigmoid_grad_conf(); }

Maybe<void> SigmoidGradOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("dx") = *GetBlobDesc4BnInOp("y");
  return Maybe<void>::Ok();
}

void SigmoidGradOp::GetSbpSignatures(
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split(input_bns(), 0)
      .Split(output_bns(), 0)
      .MakeSplitSignatureListBuilder(LogicalBlobDesc4Ibn(input_bns().Get(0)).shape().NumAxes())
      .Build(sbp_sig_list);
}

REGISTER_OP(OperatorConf::kSigmoidGradConf, SigmoidGradOp);

}  // namespace oneflow
