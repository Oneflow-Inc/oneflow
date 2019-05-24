#include "oneflow/core/operator/relu_op.h"

namespace oneflow {

void ReluOp::InitFromOpConf() {
  CHECK(op_conf().has_relu_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

const PbMessage& ReluOp::GetCustomizedConf() const { return op_conf().relu_conf(); }

void ReluOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("in");
}

void ReluOp::GetSbpSignatures(
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder().Split("in", 0).Split("out", 0).Build(
      sbp_sig_list->mutable_sbp_signature()->Add());
}

REGISTER_OP(OperatorConf::kReluConf, ReluOp);

}  // namespace oneflow
