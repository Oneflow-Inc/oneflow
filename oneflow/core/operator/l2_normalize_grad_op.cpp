#include "oneflow/core/operator/l2_normalize_grad_op.h"

namespace oneflow {

void L2NormalizeGradOp::InitFromOpConf() {
  CHECK(op_conf().has_l2_normalize_grad_conf());
  EnrollInputBn("dy");
  EnrollInputBn("y");
  EnrollInputBn("square_x_sum");
  EnrollOutputBn("dx");
}

const PbMessage& L2NormalizeGradOp::GetCustomizedConf() const {
  return op_conf().l2_normalize_grad_conf();
}

Maybe<void> L2NormalizeGradOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const L2NormalizeGradOpConf& conf = op_conf().l2_normalize_grad_conf();
  const BlobDesc* dy_blob_desc = GetBlobDesc4BnInOp("dy");
  CHECK_GE_OR_RETURN(conf.axis(), 0);
  CHECK_LT_OR_RETURN(conf.axis(), dy_blob_desc->shape().NumAxes());
  CHECK_GT_OR_RETURN(conf.epsilon(), 0);
  *GetBlobDesc4BnInOp("dx") = *dy_blob_desc;
  return Maybe<void>::Ok();
}

Maybe<void> L2NormalizeGradOp::GetSbpSignatures(SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split(input_bns(), 0)
      .Split(output_bns(), 0)
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kL2NormalizeGradConf, L2NormalizeGradOp);

}  // namespace oneflow
