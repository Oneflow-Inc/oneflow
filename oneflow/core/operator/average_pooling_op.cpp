#include "oneflow/core/operator/average_pooling_op.h"

namespace oneflow {

Pooling3DKernelConf* AveragePoolingOp::GetMutPooling3DKernelConf(
    KernelConf* kernel_conf) const {
  return kernel_conf->mutable_average_pooling_3d_conf()
      ->mutable_pooling_3d_conf();
}

}  // namespace oneflow
