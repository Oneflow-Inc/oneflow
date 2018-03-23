#include "oneflow/core/operator/normalization_model_update_op.h"

namespace oneflow {

void NormalizationModelUpdtOp::InitFromOpConf() {
  CHECK(op_conf().has_normalization_mdupdt_conf());
  EnrollInputBn("new_mean", false);
  EnrollInputBn("new_variance", false);
  EnrollOutputBn("moving_mean", false);
  EnrollOutputBn("moving_variance", false);
}

const PbMessage& NormalizationModelUpdtOp::GetCustomizedConf() const {
  return op_conf().normalization_mdupdt_conf();
}

REGISTER_OP(OperatorConf::kNormalizationMdupdtConf, NormalizationModelUpdtOp);

}  // namespace oneflow
