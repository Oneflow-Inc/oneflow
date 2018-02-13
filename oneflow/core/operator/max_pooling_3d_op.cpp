#include "oneflow/core/operator/max_pooling_3d_op.h"

namespace oneflow {

const PbMessage& MaxPooling3DOp::GetSpecialConf() const {
  return op_conf().max_pooling_3d_conf();
}

Pooling3DKernelConf* MaxPooling3DOp::GetMutPooling3DKernelConf(
    KernelConf* kernel_conf) const {
  return kernel_conf->mutable_max_pooling_3d_conf()->mutable_pooling_3d_conf();
}

REGISTER_OP(OperatorConf::kMaxPooling3DConf, MaxPooling3DOp);

}  //  namespace oneflow
