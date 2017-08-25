#include "oneflow/core/operator/copy_comm_net_op.h"

namespace oneflow {

void CopyCommNetOp::InitFromOpConf() {
  CHECK(op_conf().has_copy_comm_net_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

const PbMessage& CopyCommNetOp::GetSpecialConf() const {
  return op_conf().copy_comm_net_conf();
}

std::string CopyCommNetOp::obn2lbn(const std::string& output_bn) const {
  return kPackedBlobName;
}

std::string CopyCommNetOp::ibn2lbn(const std::string& input_bn) const {
  return kPackedBlobName;
}

REGISTER_OP(OperatorConf::kCopyCommNetConf, CopyCommNetOp);

}  // namespace oneflow
