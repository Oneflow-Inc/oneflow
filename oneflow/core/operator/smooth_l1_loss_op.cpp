#include "oneflow/core/operator/smooth_l1_loss_op.h"

namespace oneflow {

void SmoothL1LossOp::VirtualInitFromOpConf() { TODO(): }

const PbMessage& SparseSoftmaxCrossEntropyLossOp::GetCustomizedConf() const {
  return op_conf().smooth_l1_loss_conf();
}

LossKernelConf* SparseSoftmaxCrossEntropyLossOp::GetMutLossKernelConf(
    KernelConf* kernel_conf) const {
  return kernel_conf->mutable_smooth_l1_loss_conf()->mutable_loss_conf();
}

void SmoothL1LossOp::VirtualInferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, size_t* buf_size) const {
  TODO();
}

REGISTER_OP(OperatorConf::kSmoothL1LossConf, SmoothL1LossOp);

}  // namespace oneflow
