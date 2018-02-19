#include "oneflow/core/operator/max_pooling_2d_op.h"

namespace oneflow {

const PbMessage& MaxPooling2DOp::GetSpecialConf() const {
  return op_conf().max_pooling_2d_conf();
}

Pooling3DKernelConf* MaxPooling2DOp::GetMutPooling3DKernelConf(
    KernelConf* kernel_conf) const {
  return kernel_conf->mutable_max_pooling_3d_conf()->mutable_pooling_3d_conf();
}

REGISTER_OP(OperatorConf::kMaxPooling2DConf, MaxPooling2DOp);

}  //  namespace oneflow
