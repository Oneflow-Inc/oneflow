#include "oneflow/operator/copy_comm_net_op.h"
#include "oneflow/operator/operator_manager.h"

namespace oneflow {

void CopyCommNetOp::InitFromOpConf(const OperatorConf& op_conf) {
  CHECK(op_conf.has_copy_comm_net_conf());
  mut_op_conf() = op_conf;
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

const PbMessage& CopyCommNetOp::GetSpecialConf() const {
  return op_conf().copy_comm_net_conf();
}

std::string CopyCommNetOp::obn2lbn(const std::string& output_bn) const {
  return kBaledBlobName;
}

std::string CopyCommNetOp::ibn2lbn(const std::string& input_bn) const {
  return kBaledBlobName;
}

REGISTER_OP(OperatorConf::kCopyCommNetConf, CopyCommNetOp);

} // namespace oneflow
