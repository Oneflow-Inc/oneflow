#include "oneflow/core/operator/max_pooling_1d_op.h"

namespace oneflow {

const PbMessage& MaxPooling1DOp::GetSpecialConf() const {
  return op_conf().max_pooling_1d_conf();
}

Pooling3DKernelConf* MaxPooling1DOp::GetMutPooling3DKernelConf(
    KernelConf* kernel_conf) const {
  return kernel_conf->mutable_max_pooling_3d_conf()->mutable_pooling_3d_conf();
}

REGISTER_OP(OperatorConf::kMaxPooling1DConf, MaxPooling1DOp);

}  //  namespace oneflow
