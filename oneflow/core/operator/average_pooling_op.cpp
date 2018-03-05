#include "oneflow/core/operator/average_pooling_op.h"

namespace oneflow {

PoolingKernelConf* AveragePoolingOp::GetMutPoolingKernelConf(
    KernelConf* kernel_conf) const {
  return kernel_conf->mutable_average_pooling_conf()->mutable_pooling_conf();
}

}  // namespace oneflow
