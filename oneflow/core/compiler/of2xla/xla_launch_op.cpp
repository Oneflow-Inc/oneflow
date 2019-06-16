#include "absl/strings/str_cat.h"
#include "oneflow/core/compiler/of2xla/xla_launch_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void XlaLaunchOp::InitFromOpConf() {
  CHECK(op_conf().has_xla_launch_conf());
  const XlaLaunchOpConf &conf = op_conf().xla_launch_conf();
  int num_inputs = conf.in().size();
  int num_outputs = conf.out().size();
  for (int i = 0; i < num_inputs; ++i) {
    std::string in_name = absl::StrCat("in", i);
    EnrollInputBn(in_name);
  }
  for (int i = 0; i < num_outputs; ++i) {
    std::string out_name = absl::StrCat("out", i);
    EnrollOutputBn(out_name);
  }
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
