#include "operator/model_load_op.h"
#include "glog/logging.h"

namespace oneflow {
  
void ModelLoadOp::InitFromOpConf(const OperatorConf& op_conf) {
  CHECK(op_conf.has_model_load_op_conf());
  mut_op_conf() = op_conf;
  EnrollOutputBn("model", false);
}

std::string ModelLoadOp::GetValueFromPbOpConf(const std::string& k) const {
  return GetValueFromPbMessage(op_conf().model_load_op_conf(), k);
}
}
