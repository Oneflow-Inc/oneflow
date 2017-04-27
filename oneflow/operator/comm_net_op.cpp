#include "operator/comm_net_op.h"

namespace oneflow {

void CommNetOp::InitFromOpConf(const OperatorConf& op_conf) {
  CHECK(op_conf.has_comm_net_conf());
  mut_op_conf() = op_conf;
  EnrollInputBn("comm_net_in");
  CHECK(ibn2lbn_.emplace("comm_net_in", RegstDesc::kAllLbn).second);
  EnrollOutputBn("comm_net_out");
  CHECK(ibn2lbn_.emplace("comm_net_out", RegstDesc::kAllLbn).second);
}

std::string CommNetOp::obn2lbn(const std::string& input_bn) const {
  return RegstDesc::kAllLbn;
}

std::string CommNetOp::ibn2lbn(const std::string& input_bn) const {
  return RegstDesc::kAllLbn;
}


std::string BoxingOp::GetValueFromPbOpConf(const std::string& k) const {
  return GetValueFromPbMessage(op_conf().comm_net_conf(), k);
}
} // namespace oneflow
