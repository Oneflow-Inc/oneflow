#include "operator/comm_net_op.h"
#include "operator/operator_manager.h"

namespace oneflow {

void CommNetOp::InitFromOpConf(const OperatorConf& op_conf) {
  CHECK(op_conf.has_comm_net_conf());
  mut_op_conf() = op_conf;
  EnrollInputBn("comm_net_in");
  EnrollOutputBn("comm_net_out");
}

const PbMessage& CommNetOp::GetSpecialConf() const {
  return op_conf().comm_net_conf();
}

std::string CommNetOp::obn2lbn(const std::string& output_bn) const {
  return RegstDesc::kAllLbn;
}

std::string CommNetOp::ibn2lbn(const std::string& input_bn) const {
  return RegstDesc::kAllLbn;
}

REGISTER_OP(OperatorConf::kCommNetConf, CommNetOp);

} // namespace oneflow
