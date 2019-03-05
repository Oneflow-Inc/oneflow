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
  ones->mut_shape() = prediction->shape();
}

void IdentityLossOp::GenerateBackwardOpConf(
    std::vector<OperatorConf>* ops,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) const {
  auto* diff_lbi = DiffLbi4BnInOp("prediction");
  if (diff_lbi != nullptr) {
    OperatorConf mul_zero_op;
    mul_zero_op.set_name(op_name() + "_grad_stage0");
    ScalarMulOpConf* mul_zero_op_conf = mul_zero_op.mutable_scalar_mul_conf();
    mul_zero_op_conf->set_in(GenLogicalBlobName(BnInOp2Lbi("prediction")));
    mul_zero_op_conf->set_out("out");
    mul_zero_op_conf->set_int_operand(0);
    ops->push_back(mul_zero_op);

    OperatorConf add_one_op;
    add_one_op.set_name(op_name() + "_grad_stage1");
    ScalarAddOpConf* add_one_op_conf = add_one_op.mutable_scalar_add_conf();
    add_one_op_conf->set_in(mul_zero_op.name() + "/out");
    add_one_op_conf->set_out("out");
    add_one_op_conf->set_int_operand(1);
    ops->push_back(add_one_op);

    diff_lbi->set_op_name(add_one_op.name());
    diff_lbi->set_blob_name("out");
  }
}

REGISTER_OP(OperatorConf::kIdentityLossConf, IdentityLossOp);

}  // namespace oneflow
