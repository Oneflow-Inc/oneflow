#include "oneflow/core/operator/max_pooling_op.h"

namespace oneflow {

PoolingKernelConf* MaxPoolingOp::GetMutPoolingKernelConf(
    KernelConf* kernel_conf) const {
  return kernel_conf->mutable_max_pooling_conf()->mutable_pooling_conf();
}

}  // namespace oneflow
