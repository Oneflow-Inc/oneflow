#include "oneflow/core/operator/pooling_op.h"

namespace oneflow {

void PoolingOp::InitFromOpConf() {
  std::string padding_mthd = GetStringFromSpecialConf("padding");
  std::transform(padding_mthd.begin(), padding_mthd.end(), padding_mthd.begin(),
                 ::tolower);
  if (padding_mthd != "same" && padding_mthd != "valid") {
    LOG(FATAL) << "Invalid padding method in " << op_name();
  }
  SetStringInSpecialConf("padding", padding_mthd);
  VirtualCheckPoolSizeAndStrides();
  EnrollInputBn("in");
  EnrollOutputBn("out");
  VirtualEnrollDataTmpBn();
}

}  // namespace oneflow
