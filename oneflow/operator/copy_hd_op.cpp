#include "oneflow/operator/copy_hd_op.h"
#include "oneflow/operator/operator_manager.h"

namespace oneflow {

void CopyHdOp::InitFromOpConf(const OperatorConf& op_conf) {
  CHECK(op_conf.has_copy_hd_conf());
  mut_op_conf() = op_conf;
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

const PbMessage& CopyHdOp::GetSpecialConf() const {
  return op_conf().copy_hd_conf();
}

REGISTER_OP(OperatorConf::kCopyHdConf, CopyHdOp);

} // namespace oneflow
