#include "oneflow/core/operator/conv_3d_op.h"

namespace oneflow {

const PbMessage& Conv3DOp::GetSpecialConf() const {
  return op_conf().conv_3d_conf();
}

int32_t Conv3DOp::ModelSplitAxis() const {
  if (GetStringFromSpecialConf("data_format") == "channel_first") {
    return 1;
  } else {
    return 4;
  }
}

PbMessage* Conv3DOp::MutableConvKernelConf(KernelConf* kernel_conf) {
  return kernel_conf->mutable_conv_3d_conf();
}

REGISTER_OP(OperatorConf::kConv3DConf, Conv3DOp);

}  // namespace oneflow
