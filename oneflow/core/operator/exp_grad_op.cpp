#include "oneflow/core/operator/exp_grad_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void ExpGradOp::InitFromOpConf() {
  CHECK(op_conf().has_exp_grad_conf());
  EnrollInputBn("dy");
  EnrollInputBn("y");
  EnrollOutputBn("dx")->set_mutable_inplace_ibn("dy");
}

const PbMessage& ExpGradOp::GetCustomizedConf() const { return op_conf().exp_grad_conf(); }

void ExpGradOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                               const ParallelContext* parallel_ctx) const {
  const BlobDesc* dy = GetBlobDesc4BnInOp("dy");
  CHECK(*dy == *GetBlobDesc4BnInOp("y"));
  *GetBlobDesc4BnInOp("dx") = *dy;
}

void ExpGradOp::GetSbpSignatures(
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split(input_bns(), 0)
      .Split(output_bns(), 0)
      .MakeSplitSignatureListBuilder(LogicalBlobDesc4Ibn(output_bns().Get(0)).shape().NumAxes())
      .Build(sbp_sig_list);
  SbpSignatureBuilder().Broadcast("y").PartialSum("dy").PartialSum("dx").Build(
      sbp_sig_list->mutable_sbp_signature()->Add());
}

REGISTER_OP(OperatorConf::kExpGradConf, ExpGradOp);

}  // namespace oneflow
