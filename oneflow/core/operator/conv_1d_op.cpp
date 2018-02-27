#include "oneflow/core/operator/conv_1d_op.h"

namespace oneflow {

const PbMessage& Conv1DOp::GetSpecialConf() const {
  return op_conf().conv_1d_conf();
}

int32_t Conv1DOp::ModelSplitAxis() const {
  if (GetStringFromSpecialConf("data_format") == "channel_first") {
    return 1;
  } else {
    return 2;
  }
}

PbMessage* Conv1DOp::MutableConvKernelConf(KernelConf* kernel_conf) {
  return kernel_conf->mutable_conv_1d_conf();
}

REGISTER_OP(OperatorConf::kConv1DConf, Conv1DOp);

}  // namespace oneflow
