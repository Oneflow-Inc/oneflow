#include "oneflow/core/operator/model_save_op.h"

namespace oneflow {

void ModelSaveOp::InitFromOpConf() { CHECK(op_conf().has_model_save_conf()); }

const PbMessage& ModelSaveOp::GetCustomizedConf() const {
  return op_conf().model_save_conf();
}

REGISTER_OP(OperatorConf::kModelSaveConf, ModelSaveOp);

}  // namespace oneflow
