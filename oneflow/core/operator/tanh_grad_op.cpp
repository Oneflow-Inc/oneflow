#include "oneflow/core/operator/tanh_grad_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void TanHGradOp::InitFromOpConf() {
  CHECK(op_conf().has_tanh_grad_conf());
  EnrollInputBn("y");
  EnrollInputBn("dy");
  EnrollOutputBn("dx")->set_mutable_inplace_ibn("dy");
}

const PbMessage& TanHGradOp::GetCustomizedConf() const { return op_conf().tanh_grad_conf(); }

Maybe<void> TanHGradOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("dx") = *GetBlobDesc4BnInOp("y");
  return Maybe<void>::Ok();
}

Maybe<void> TanHGradOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split(input_bns(), 0)
      .Split(output_bns(), 0)
      .MakeSplitSignatureListBuilder(JUST(LogicalBlobDesc4Ibn("y"))->shape().NumAxes())
      .Build(sbp_sig_list);
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kTanhGradConf, TanHGradOp);

}  // namespace oneflow
