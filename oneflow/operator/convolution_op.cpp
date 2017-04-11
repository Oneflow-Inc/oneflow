#include "operator/convolution_op.h"
#include "glog/logging.h"

namespace oneflow {

void ConvolutionOp::Init(const OperatorConf& op_conf) {
  mut_op_name() = op_conf.name();
  
  CHECK(op_conf.has_convolution_op_conf());
  auto cnf = new ConvolutionOpConf(op_conf.convolution_op_conf());
  mut_pb_op_conf().reset(cnf);
  
  EnrollInputBn("in");
  EnrollInputDiffBn(GenDiffBn("in"));
  EnrollOutputBn("out");
  EnrollOutputDiffBn(GenDiffBn("out"));
  EnrollDataTmpBn("col_buf");
  
  EnrollModelBn("weight");
  EnrollModelDiffBn(GenDiffBn("weight"));
  EnrollModelBn("bias");
  EnrollModelDiffBn(GenDiffBn("bias"));
  EnrollModelTmpBn("bias_multiplier");
}

} // namespace oneflow
