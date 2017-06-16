#include "oneflow/core/operator/gpu_clear_op.h"
#include "oneflow/core/operator/operator_manager.h"

namespace oneflow {

void GpuClearOp::InitFromOpConf(const OperatorConf& op_conf) {
  CHECK(op_conf.has_gpu_clear_conf());
  mut_op_conf() = op_conf;

  EnrollInputBn("in");
}

const PbMessage& GpuClearOp::GetSpecialConf() const {
  return op_conf().gpu_clear_conf();
}

REGISTER_OP(OperatorConf::kGpuClearConf, GpuClearOp);

}  // namespace oneflow
