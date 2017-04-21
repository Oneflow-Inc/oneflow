#include "operator/model_save_op.h"
#include "glog/logging.h"

namespace oneflow {

void ModelSaveOp::InitFromOpConf(const OperatorConf& op_conf) {
  CHECK(op_conf.has_model_save_op_conf());
  mut_op_conf() = op_conf;
  EnrollInputBn("model", false);
}

std::string ModelSaveOp::GetValueFromPbOpConf(const std::string& k) const {
  return GetValueFromPbMessage(op_conf().model_save_op_conf(), k);
}
}
