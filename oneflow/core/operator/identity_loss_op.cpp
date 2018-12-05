#include "oneflow/core/operator/identity_loss_op.h"

namespace oneflow {

const PbMessage& IdentityLossOp::GetCustomizedConf() const {
  return op_conf().identity_loss_conf();
}

LossKernelConf* IdentityLossOp::GetMutLossKernelConf(KernelConf* kernel_conf) const {
  return kernel_conf->mutable_identity_loss_conf()->mutable_loss_conf();
}

void IdentityLossOp::VirtualInitFromOpConf() { EnrollConstBufBn("ones"); }

void IdentityLossOp::VirtualInferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* prediction = GetBlobDesc4BnInOp("prediction");
  BlobDesc* ones = GetBlobDesc4BnInOp("ones");
  ones->set_data_type(prediction->data_type());
  ones->mut_shape() = ones->shape();
}

REGISTER_OP(OperatorConf::kIdentityLossConf, IdentityLossOp);

}  // namespace oneflow
