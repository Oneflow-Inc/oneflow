#include "operator/copy_hd_op.h"
#include "operator/operator_manager.h"

namespace oneflow {

void CopyHdOp::InitFromOpConf(const OperatorConf& op_conf) {
  CHECK(op_conf.has_copy_hd_conf());
  mut_op_conf() = op_conf;
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

std::string CopyHdOp::GetValueFromPbOpConf(const std::string& k) const {
  return GetValueFromPbMessage(op_conf().copy_hd_conf(), k);
}

REGISTER_OP(OperatorConf::kCopyHdConf, CopyHdOp);

} // namespace oneflow
