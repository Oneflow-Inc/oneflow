#include "operator/model_load_op.h"
#include "glog/logging.h"
#include "operator/operator_manager.h"

namespace oneflow {
  
void ModelLoadOp::InitFromOpConf(const OperatorConf& op_conf) {
  CHECK(op_conf.has_model_load_conf());
  mut_op_conf() = op_conf;
  EnrollOutputBn("model", false);
}

const PbMessage& ModelLoadOp::GetSpecialConf() const {
  return op_conf().model_load_conf();
}

REGISTER_OP(OperatorConf::kModelLoadConf, ModelLoadOp);

}
