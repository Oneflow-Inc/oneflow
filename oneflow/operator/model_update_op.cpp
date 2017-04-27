#include "operator/model_update_op.h"
#include "glog/logging.h"
#include "operator/operator_manager.h"

namespace oneflow {

void ModelUpdateOp::InitFromOpConf(const OperatorConf& op_conf) {
  CHECK(op_conf.has_model_update_conf());
  mut_op_conf() = op_conf;
  
  EnrollInputBn("model_diffs", false);
  EnrollInputBn("model_init", false);
  EnrollOutputBn("model", false);
}

std::string ModelUpdateOp::GetValueFromPbOpConf(const std::string& k) const {
  return GetValueFromPbMessage(op_conf().model_update_conf(), k);
}

REGISTER_OP(OperatorConf::kModelUpdateConf, ModelUpdateOp);

}
