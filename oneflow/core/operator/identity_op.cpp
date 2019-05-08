#include "oneflow/core/operator/identity_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void IdentityOp::InitFromOpConf() {
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

void IdentityOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("in");
}

const PbMessage& IdentityOp::GetCustomizedConf() const { return op_conf().identity_conf(); }

void IdentityOp::GetSbpSignatures(
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  const auto bns = StdVec2PbRpf<std::string>({"in", "out"});
  SbpSignatureBuilder().Broadcast(bns).Build(sbp_sig_list->mutable_sbp_signature()->Add());
  SbpSignatureBuilder().PartialSum(bns).Build(sbp_sig_list->mutable_sbp_signature()->Add());
  FOR_RANGE(int64_t, i, 0, LogicalBlobDesc4Ibn("in").shape().NumAxes()) {
    SbpSignatureBuilder().Split(bns, i).Build(sbp_sig_list->mutable_sbp_signature()->Add());
  }
}

REGISTER_OP(OperatorConf::kIdentityConf, IdentityOp);

}  // namespace oneflow
