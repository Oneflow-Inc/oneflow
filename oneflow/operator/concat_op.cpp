#include "operator/concat_op.h"

namespace oneflow {

void ConcatOp::InitFromOpConf(const OperatorConf& op_conf) {
  CHECK(op_conf.has_concat_conf());
  mut_op_conf() = op_conf;

  for (int i = 0; i < op_conf.concat_conf().in_size(); ++i) {
    EnrollInputBn("in_" + std::to_string(i));
  }
  EnrollOutputBn("out");
}
std::string ConcatOp::GetValueFromPbOpConf(const std::string& k) const {
  return GetValueFromPbMessage(op_conf().concat_conf(), k);
}

} // namespace oneflow
