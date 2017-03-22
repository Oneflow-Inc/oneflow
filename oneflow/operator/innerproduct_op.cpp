#include "operator/innerproduct_op.h"
#include "glog/logging.h"

namespace oneflow {

void InnerProductOp::Init(const OperatorConf& op_conf) {
  mut_op_name() = op_conf.name();
  
  CHECK(op_conf.has_inner_product_op_conf());
  auto cnf = new InnerProductOpConf(op_conf.inner_product_op_conf());
  mut_pb_op_conf().reset(cnf);
  
  RegisterInputBlobName("in");
  RegisterInputDiffBlobName(GenDiffBlobName("in"));
  RegisterOutputBlobName("out");
  RegisterOutputDiffBlobName(GenDiffBlobName("out"));
  
  RegisterModelBlobName("weight");
  RegisterModelDiffBlobName(GenDiffBlobName("weight"));
  RegisterModelBlobName("bias");
  RegisterModelDiffBlobName(GenDiffBlobName("bias"));
  RegisterModelTmpBlobName("bias_multiplier");
}

} // namespace oneflow
