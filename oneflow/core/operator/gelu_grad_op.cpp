#include "oneflow/core/operator/gelu_grad_op.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void GeluGradOp::InitFromOpConf() {
  CHECK(op_conf().has_gelu_grad_conf());
  EnrollInputBn("x");
  EnrollInputBn("dy");
  EnrollOutputBn("dx");
}

const PbMessage& GeluGradOp::GetCustomizedConf() const { return op_conf().gelu_grad_conf(); }

Maybe<void> GeluGradOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("dx") = *GetBlobDesc4BnInOp("x");
  return Maybe<void>::Ok();
}

void GeluGradOp::GetSbpSignatures(SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split(input_bns(), 0)
      .Split(output_bns(), 0)
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
}

REGISTER_OP(OperatorConf::kGeluGradConf, GeluGradOp);

}  // namespace oneflow
