#include "operator/clone_op.h"

namespace oneflow {

void CloneOp::InitFromOpConf(const OperatorConf& op_conf) {
  CHECK(op_conf.has_clone_op_conf());
  mut_op_conf() = op_conf;
  EnrollInputBn("in");
  for (int64_t i = 0; i < op_conf.clone_op_conf().out_num(); ++i) {
    EnrollOutputBn("out_" + std::to_string(i));
  }
}
std::string CloneOp::GetValueFromPbOpConf(const std::string& k) const {
  return GetValueFromPbMessage(op_conf().clone_op_conf(), k);
}
} // namespace oneflow
