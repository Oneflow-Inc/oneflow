#include "operator/convolution_op.h"
#include "glog/logging.h"

namespace oneflow {

void ConvolutionOp::InitFromOpConf(const OperatorConf& op_conf) {
  CHECK(op_conf.has_convolution_conf());
  mut_op_conf() = op_conf;
  
  EnrollInputBn("in");
  EnrollOutputBn("out");
  EnrollDataTmpBn("col_buf");
  
  EnrollModelBn("weight");
  EnrollModelBn("bias");
  EnrollModelTmpBn("bias_multiplier");
}

std::string ConvolutionOp::GetValueFromPbOpConf(const std::string& k) const {
  return GetValueFromPbMessage(op_conf().convolution_conf(), k);
}
} // namespace oneflow
