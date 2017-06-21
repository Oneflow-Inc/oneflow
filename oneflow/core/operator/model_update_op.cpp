#include "oneflow/core/operator/model_update_op.h"

namespace oneflow {

void ModelUpdateOp::InitFromOpConf(const OperatorConf& op_conf) {
  CHECK(op_conf.has_model_update_conf());
  mut_op_conf() = op_conf;
  
  EnrollInputBn("model_diffs", false);
  EnrollOutputBn("model", false);
}

const PbMessage& ModelUpdateOp::GetSpecialConf() const {
  return op_conf().model_update_conf();
}

REGISTER_OP(OperatorConf::kModelUpdateConf, ModelUpdateOp);

}
