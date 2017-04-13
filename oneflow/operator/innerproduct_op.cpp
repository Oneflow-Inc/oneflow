#include "operator/innerproduct_op.h"
#include "glog/logging.h"

namespace oneflow {

void InnerProductOp::Init(const OperatorConf& op_conf) {
  mut_op_name() = op_conf.name();
  
  CHECK(op_conf.has_inner_product_op_conf());
  auto cnf = new InnerProductOpConf(op_conf.inner_product_op_conf());
  mut_pb_op_conf().reset(cnf);
  
  EnrollInputBn("in");
  EnrollOutputBn("out");
  
  EnrollModelBn("weight");
  EnrollModelBn("bias");
  EnrollModelTmpBn("bias_multiplier");
}

} // namespace oneflow
