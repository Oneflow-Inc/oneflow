#include "oneflow/core/operator/model_save_op.h"

namespace oneflow {

void ModelSaveOp::InitFromOpConf(const OperatorConf& op_conf) {
  CHECK(op_conf.has_model_save_conf());
  mut_op_conf() = op_conf;
  for (const std::string& lbn : op_conf.model_save_conf().lbns()) {
    EnrollInputBn("in_" + lbn);
  }
}

const PbMessage& ModelSaveOp::GetSpecialConf() const {
  return op_conf().model_save_conf();
}

REGISTER_OP(OperatorConf::kModelSaveConf, ModelSaveOp);

}  // namespace oneflow
