#include "operator/convolution_op.h"
#include "glog/logging.h"

namespace oneflow {

void ConvolutionOp::Init(const OperatorConf& op_conf) {
  mut_op_name() = op_conf.name();
  
  CHECK(op_conf.has_convolution_op_conf());
  auto cnf = new ConvolutionOpConf(op_conf.convolution_op_conf());
  mut_pb_op_conf().reset(cnf);
  
  RegisterInputBlobName("in");
  RegisterInputDiffBlobName(GenDiffBlobName("in"));
  RegisterOutputBlobName("out");
  RegisterOutputDiffBlobName(GenDiffBlobName("out"));
  RegisterDataTmpBlobName("col_buf");
  
  RegisterModelBlobName("weight");
  RegisterModelDiffBlobName(GenDiffBlobName("weight"));
  RegisterModelBlobName("bias");
  RegisterModelDiffBlobName(GenDiffBlobName("bias"));
  RegisterModelTmpBlobName("bias_multiplier");
}

} // namespace oneflow
