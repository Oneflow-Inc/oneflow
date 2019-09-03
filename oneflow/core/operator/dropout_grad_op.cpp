#include "oneflow/core/operator/dropout_grad_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void DropoutGradOp::InitFromOpConf() {
  double dropout_rate = op_conf().dropout_conf().rate();
  CHECK_GE(dropout_rate, 0);
  CHECK_LT(dropout_rate, 1);
  EnrollInputBn("dy");
  EnrollInputBn("random_mask");
  EnrollOutputBn("dx");
}

const PbMessage& DropoutGradOp::GetCustomizedConf() const { return op_conf().dropout_grad_conf(); }

Maybe<void> DropoutGradOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  BlobDesc* dy_desc = GetBlobDesc4BnInOp("dy");
  *GetBlobDesc4BnInOp("dx") = *dy_desc;
  CHECK_EQ_OR_RETURN(dy_desc->shape(), GetBlobDesc4BnInOp("random_mask")->shape());
  return Maybe<void>::Ok();
}

Maybe<void> DropoutGradOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split(input_bns(), 0)
      .Split(output_bns(), 0)
      .MakeSplitSignatureListBuilder(JUST(LogicalBlobDesc4Ibn("dy"))->shape().NumAxes())
      .Build(sbp_sig_list);
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kDropoutGradConf, DropoutGradOp);

}  // namespace oneflow
