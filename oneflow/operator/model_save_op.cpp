#include "operator/model_save_op.h"
#include "glog/logging.h"
#include "operator/operator_manager.h"

namespace oneflow {

void ModelSaveOp::InitFromOpConf(const OperatorConf& op_conf) {
  CHECK(op_conf.has_model_save_conf());
  mut_op_conf() = op_conf;
  EnrollInputBn("model", false);
}

const PbMessage& ModelSaveOp::GetSpecialConf() const {
  return op_conf().model_save_conf();
}

REGISTER_OP(OperatorConf::kModelSaveConf, ModelSaveOp);

}
