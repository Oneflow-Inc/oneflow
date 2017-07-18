#include "oneflow/core/operator/rmsprop_model_update_op.h"

namespace oneflow {

void RMSPropModelUpdateOp::InitFromOpConf(const OperatorConf& op_conf) {
  CHECK(op_conf.has_rmsprop_mdupdt_conf());
  mut_op_conf() = op_conf;

  EnrollInputBn("model_diffs", false);
  EnrollModelTmpBn("mean_square");
  EnrollOutputBn("model", false);
}

const PbMessage& RMSPropModelUpdateOp::GetSpecialConf() const {
  return op_conf().rmsprop_mdupdt_conf();
}

REGISTER_OP(OperatorConf::kRmspropMdupdtConf, RMSPropModelUpdateOp);

}  // namespace oneflow
