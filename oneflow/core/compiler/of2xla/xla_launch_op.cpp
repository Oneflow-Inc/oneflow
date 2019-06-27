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

void XlaLaunchOp::InferSbpSignature(
    SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
    const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
    std::function<const SbpInferHint&(const std::string&)> SbpInferHint4Ibn,
    const ParallelDesc& parallel_desc) const {
  const auto &xla_launch_conf = op_conf().xla_launch_conf();
  CHECK(xla_launch_conf.has_sbp_signature());
  const SbpSignature &sbp_conf = xla_launch_conf.sbp_signature();
  this->InferSbpSignature(sbp_signature, sbp_conf, CalcOrderValue4SbpSig,
                          SbpInferHint4Ibn, parallel_desc);
}

REGISTER_OP(OperatorConf::kXlaLaunchConf, XlaLaunchOp);

}  // namespace oneflow
