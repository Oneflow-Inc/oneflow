#include "operator/softmax_op.h"
#include "glog/logging.h"
#include "operator/operator_factory.h"

namespace oneflow {

void SoftmaxOp::InitFromOpConf(const OperatorConf& op_conf) {
  CHECK(op_conf.has_softmax_conf());
  mut_op_conf() = op_conf;

  EnrollInputBn("in");
  EnrollOutputBn("out");
}

std::string SoftmaxOp::GetValueFromPbOpConf(const std::string& k) const {
  return GetValueFromPbMessage(op_conf().softmax_conf(), k);
}

REGISTER_OP(OperatorConf::kSoftmaxConf, SoftmaxOp);

} // namespace oneflow
