#include "oneflow/core/operator/average_pooling_2d_op.h"

namespace oneflow {

const PbMessage& AveragePooling2DOp::GetSpecialConf() const {
  return op_conf().average_pooling_2d_conf();
}

PoolingKernelConf* AveragePooling2DOp::GetMutPoolingKernelConf(
    KernelConf* kernel_conf) const {
  return kernel_conf->mutable_average_pooling_conf()->mutable_pooling_conf();
}

REGISTER_OP(OperatorConf::kAveragePooling2DConf, AveragePooling2DOp);

}  // namespace oneflow
