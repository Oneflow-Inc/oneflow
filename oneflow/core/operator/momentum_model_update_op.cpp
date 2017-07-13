#include "oneflow/core/operator/momentum_model_update_op.h"

namespace oneflow {

void MomentumModelUpdateOp::InitFromOpConf(const OperatorConf& op_conf) {
  CHECK(op_conf.has_momentum_model_update_conf());
  mut_op_conf() = op_conf;

  EnrollInputBn("model_diffs", false);
  EnrollInputBn("momentum", false);
  EnrollOutputBn("model", false);
}

const PbMessage& MomentumModelUpdateOp::GetSpecialConf() const {
  return op_conf().momentum_model_update_conf();
}

REGISTER_OP(OperatorConf::kMomentumModelUpdateConf, MomentumModelUpdateOp);

}  // namespace oneflow
