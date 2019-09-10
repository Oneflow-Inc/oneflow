#include "oneflow/core/operator/gelu_op.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void GeluOp::InitFromOpConf() {
  CHECK(op_conf().has_gelu_conf());

  EnrollInputBn("in");
  EnrollOutputBn("out");
}

const PbMessage& GeluOp::GetCustomizedConf() const { return op_conf().gelu_conf(); }

Maybe<void> GeluOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                   const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("in");
  return Maybe<void>::Ok();
}

Maybe<void> GeluOp::GetSbpSignatures(SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split(input_bns(), 0)
      .Split(output_bns(), 0)
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kGeluConf, GeluOp);

}  // namespace oneflow
