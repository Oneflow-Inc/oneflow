#include "oneflow/core/operator/average_pooling_3d_op.h"

namespace oneflow {

const PbMessage& AveragePooling3DOp::GetSpecialConf() const {
  return op_conf().average_pooling_3d_conf();
}

Pooling3DKernelConf* AveragePooling3DOp::GetMutPooling3DKernelConf(
    KernelConf* kernel_conf) const {
  return kernel_conf->mutable_average_pooling_3d_conf()
      ->mutable_pooling_3d_conf();
}

REGISTER_OP(OperatorConf::kAveragePooling3DConf, AveragePooling3DOp);

}  // namespace oneflow
