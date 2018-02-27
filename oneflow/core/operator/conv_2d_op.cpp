#include "oneflow/core/operator/conv_2d_op.h"

namespace oneflow {

const PbMessage& Conv2DOp::GetSpecialConf() const {
  return op_conf().conv_2d_conf();
}

int32_t Conv2DOp::ModelSplitAxis() const {
  if (GetStringFromSpecialConf("data_format") == "channel_first") {
    return 1;
  } else {
    return 3;
  }
}
PbMessage* Conv2DOp::MutableConvKernelConf(KernelConf* kernel_conf) {
  return kernel_conf->mutable_conv_2d_conf();
}

REGISTER_OP(OperatorConf::kConv2DConf, Conv2DOp);

}  // namespace oneflow
