#include "operator/clone_op.h"
#include "operator/operator_factory.h"

namespace oneflow {

void CloneOp::InitFromOpConf(const OperatorConf& op_conf) {
  CHECK(op_conf.has_clone_conf());
  mut_op_conf() = op_conf;
  
  EnrollInputBn("in");
  for (int64_t i = 0; i < op_conf.clone_conf().out_num(); ++i) {
    EnrollOutputBn("out_" + std::to_string(i));
  }
}
std::string CloneOp::GetValueFromPbOpConf(const std::string& k) const {
  return GetValueFromPbMessage(op_conf().clone_conf(), k);
}

REGISTER_OP(OperatorConf::kCloneConf, CloneOp);

} // namespace oneflow
