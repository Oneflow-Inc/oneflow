#include "oneflow/core/operator/copy_hd_op.h"

namespace oneflow {

void CopyHdOp::InitFromOpConf() {
  CHECK(op_conf().has_copy_hd_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

const PbMessage& CopyHdOp::GetSpecialConf() const {
  return op_conf().copy_hd_conf();
}

REGISTER_OP(OperatorConf::kCopyHdConf, CopyHdOp);

}  // namespace oneflow
