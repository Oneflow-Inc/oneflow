#include "oneflow/core/operator/average_pooling_2d_op.h"

namespace oneflow {

const PbMessage& AveragePooling2DOp::GetSpecialConf() const {
  return op_conf().average_pooling_2d_conf();
}

Pooling2DKernelConf* AveragePooling2DOp::GetMutPooling2DKernelConf(
    KernelConf* kernel_conf) const {
  return kernel_conf->mutable_average_pooling_2d_conf()
      ->mutable_pooling_2d_conf();
}

REGISTER_OP(OperatorConf::kAveragePooling2DConf, AveragePooling2DOp);

}  // namespace oneflow
