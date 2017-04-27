#include "operator/concat_op.h"
#include "operator/operator_manager.h"

namespace oneflow {

void ConcatOp::InitFromOpConf(const OperatorConf& op_conf) {
  CHECK(op_conf.has_concat_conf());
  mut_op_conf() = op_conf;

  for (int i = 0; i < op_conf.concat_conf().in_size(); ++i) {
    std::string ibn = "in_" + std::to_string(i);
    EnrollInputBn(ibn);
    CHECK(ibn2lbn_.emplace(ibn, op_conf.concat_conf().in(i)).second);
  }
  EnrollOutputBn("out");
}
std::string ConcatOp::GetValueFromPbOpConf(const std::string& k) const {
  return GetValueFromPbMessage(op_conf().concat_conf(), k);
}

REGISTER_OP(OperatorConf::kConcatConf, ConcatOp);

} // namespace oneflow
