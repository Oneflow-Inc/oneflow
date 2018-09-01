#include "oneflow/core/operator/copy_local_op.h"

namespace oneflow {

void CopyLocalOp::InitFromOpConf() {
  EnrollInputBn("in", false);
  EnrollOutputBn("out", false);
}

const PbMessage& CopyLocalOp::GetCustomizedConf() const { return op_conf().copy_local_conf(); }

REGISTER_OP(OperatorConf::kCopyLocalConf, CopyLocalOp);

}  // namespace oneflow
