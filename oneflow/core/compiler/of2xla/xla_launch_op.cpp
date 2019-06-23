#include "oneflow/core/compiler/of2xla/xla_launch_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void XlaLaunchOp::InitFromOpConf() {
  CHECK(op_conf().has_xla_launch_conf());

  EnrollRepeatedInputBn("in");
  EnrollRepeatedOutputBn("out");
}

const PbMessage &XlaLaunchOp::GetCustomizedConf() const {
  return op_conf().xla_launch_conf();
}

void XlaLaunchOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {

}

void XlaLaunchOp::GetSbpSignatures(
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {

}

REGISTER_OP(OperatorConf::kXlaLaunchConf, XlaLaunchOp);

}  // namespace oneflow
