#include "oneflow/core/operator/average_pooling_1d_op.h"

namespace oneflow {

const PbMessage& AveragePooling1DOp::GetSpecialConf() const {
  return op_conf().average_pooling_1d_conf();
}

Pooling3DKernelConf* AveragePooling1DOp::GetMutPooling3DKernelConf(
    KernelConf* kernel_conf) const {
  return kernel_conf->mutable_average_pooling_3d_conf()
      ->mutable_pooling_3d_conf();
}

REGISTER_OP(OperatorConf::kAveragePooling1DConf, AveragePooling1DOp);

}  // namespace oneflow
