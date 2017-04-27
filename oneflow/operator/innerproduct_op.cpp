#include "operator/innerproduct_op.h"
#include "glog/logging.h"
#include "operator/operator_manager.h"

namespace oneflow {

void InnerProductOp::InitFromOpConf(const OperatorConf& op_conf) {
  CHECK(op_conf.has_innerproduct_conf());
  mut_op_conf() = op_conf;
  
  EnrollInputBn("in");
  EnrollOutputBn("out");
  
  EnrollModelBn("weight");
  EnrollModelBn("bias");
  EnrollModelTmpBn("bias_multiplier");
}

std::string InnerProductOp::GetValueFromPbOpConf(const std::string& k) const {
  return GetValueFromPbMessage(op_conf().innerproduct_conf(), k);
}

REGISTER_OP(OperatorConf::kInnerproductConf, InnerProductOp);

} // namespace oneflow
