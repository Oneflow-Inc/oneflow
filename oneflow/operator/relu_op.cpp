#include "operator/relu_op.h"
#include "glog/logging.h"
#include "operator/operator_factory.h"

namespace oneflow {

void ReluOp::InitFromOpConf(const OperatorConf& op_conf) {
  CHECK(op_conf.has_relu_conf());
  mut_op_conf() = op_conf;

  EnrollInputBn("in");
  EnrollOutputBn("out");
}

std::string ReluOp::GetValueFromPbOpConf(const std::string& k) const {
  return GetValueFromPbMessage(op_conf().relu_conf(), k);
}

REGISTER_OP(OperatorConf::kReluConf, ReluOp);

} // namespace oneflow
