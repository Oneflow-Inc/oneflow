#include "oneflow/core/operator/normal_model_update_op.h"

namespace oneflow {

void NormalModelUpdateOp::InitFromOpConf() {
  EnrollInputBn("model_diff_acc", false);
  EnrollOutputBn("model", false);
}

const PbMessage& NormalModelUpdateOp::GetSpecialConf() const {
  return op_conf().normal_mdupdt_conf();
}

REGISTER_OP(OperatorConf::kNormalMdupdtConf, NormalModelUpdateOp);

}  // namespace oneflow
