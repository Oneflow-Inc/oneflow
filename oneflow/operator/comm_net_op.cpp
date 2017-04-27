#include "operator/comm_net_op.h"

namespace oneflow {

void CommNetOp::InitFromOpConf(const OperatorConf& op_conf) {
  CHECK(op_conf.has_comm_net_conf());
  mut_op_conf() = op_conf;
}

std::string CommNetOp::obn2lbn(const std::string& input_bn) const {
  return oneflow::RegstDesc::kAllLbn;
}

std::string CommNetOp::ibn2lbn(const std::string& input_bn) const {
  return oneflow::RegstDesc::kAllLbn;
}


std::string BoxingOp::GetValueFromPbOpConf(const std::string& k) const {
  return GetValueFromPbMessage(op_conf().comm_net_conf(), k);
}
} // namespace oneflow
