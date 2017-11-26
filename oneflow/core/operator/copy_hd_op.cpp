#include "oneflow/core/operator/copy_hd_op.h"

namespace oneflow {

void CopyHdOp::InitFromOpConf() {
  EnrollInputBn("in", false);
  EnrollOutputBn("out", false);
}

const PbMessage& CopyHdOp::GetSpecialConf() const {
  return op_conf().copy_hd_conf();
}

REGISTER_OP(OperatorConf::kCopyHdConf, CopyHdOp);

}  // namespace oneflow
