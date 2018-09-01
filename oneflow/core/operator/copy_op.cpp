#include "oneflow/core/operator/copy_op.h"

namespace oneflow {

void CopyOp::InitFromOpConf() {
  EnrollInputBn("in", false);
  EnrollOutputBn("out", false);
}

const PbMessage& CopyOp::GetCustomizedConf() const { return op_conf().copy_conf(); }

REGISTER_OP(OperatorConf::kCopyConf, CopyOp);

}  // namespace oneflow
