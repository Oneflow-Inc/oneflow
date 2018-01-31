#include "oneflow/core/operator/average_pooling_op.h"

namespace oneflow {

const PbMessage& AveragePoolingOp::GetSpecialConf() const {
  return op_conf().average_pooling_conf();
}

PoolingKernelConf* AveragePoolingOp::mut_pooling_conf(
    KernelConf* kernel_conf) const {
  return kernel_conf->mutable_average_pooling_conf()->mutable_pooling_conf();
}

REGISTER_OP(OperatorConf::kAveragePoolingConf, AveragePoolingOp);

}  // namespace oneflow
