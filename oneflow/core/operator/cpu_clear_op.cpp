#include <string>
#include "oneflow/core/operator/cpu_clear_op.h"
#include "oneflow/core/operator/operator_manager.h"

namespace oneflow {

void CpuClearOp::InitFromOpConf(const OperatorConf& op_conf) {
  CHECK(op_conf.has_cpu_clear_conf());
  mut_op_conf() = op_conf;

  EnrollInputBn("in");
}

const PbMessage& CpuClearOp::GetSpecialConf() const {
  return op_conf().cpu_clear_conf();
}

REGISTER_OP(OperatorConf::kCpuClearConf, CpuClearOp);

}  // namespace oneflow
