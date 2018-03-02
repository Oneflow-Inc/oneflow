#include "oneflow/core/operator/max_pooling_op.h"

namespace oneflow {

Pooling3DKernelConf* MaxPoolingOp::GetMutPooling3DKernelConf(
    KernelConf* kernel_conf) const {
  return kernel_conf->mutable_max_pooling_3d_conf()->mutable_pooling_3d_conf();
}

}  // namespace oneflow
