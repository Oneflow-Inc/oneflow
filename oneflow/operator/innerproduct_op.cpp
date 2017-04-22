#include "operator/innerproduct_op.h"
#include "glog/logging.h"

namespace oneflow {

void InnerProductOp::InitFromOpConf(const OperatorConf& op_conf) {
  CHECK(op_conf.has_inner_product_op_conf());
  mut_op_conf() = op_conf;
  
  EnrollInputBn("in");
  EnrollOutputBn("out");
  
  EnrollModelBn("weight");
  EnrollModelBn("bias");
  EnrollModelTmpBn("bias_multiplier");
}

std::string InnerProductOp::GetValueFromPbOpConf(const std::string& k) const {
  return GetValueFromPbMessage(op_conf().inner_product_op_conf(), k);
}
} // namespace oneflow
