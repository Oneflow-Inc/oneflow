#include <string>
#include "oneflow/core/operator/clear_op.h"
#include "oneflow/core/operator/operator_manager.h"

namespace oneflow {

void ClearOp::InitFromOpConf(const OperatorConf& op_conf) {
  CHECK(op_conf.has_clear_conf());
  mut_op_conf() = op_conf;
}

const PbMessage& ClearOp::GetSpecialConf() const {
  return op_conf().clone_conf();
}

REGISTER_OP(OperatorConf::kClearConf, ClearOp);

}  // namespace oneflow
