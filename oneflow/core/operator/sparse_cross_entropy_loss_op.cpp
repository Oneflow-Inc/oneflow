#include "oneflow/core/operator/sparse_cross_entropy_loss_op.h"

namespace oneflow {

const PbMessage& SparseCrossEntropyLossOp::GetCustomizedConf() const {
  return op_conf().sparse_cross_entropy_loss_conf();
}

LossKernelConf* SparseCrossEntropyLossOp::GetMutLossKernelConf(KernelConf* kernel_conf) const {
  return kernel_conf->mutable_sparse_cross_entropy_loss_conf()->mutable_loss_conf();
}

REGISTER_OP(OperatorConf::kSparseCrossEntropyLossConf, SparseCrossEntropyLossOp);

}  // namespace oneflow
