#include "oneflow/core/operator/hinge_loss_op.h"

namespace oneflow {

void HingeLossOp::VirtualInitFromOpConf() { EnrollDataTmpBn("tmp_diff"); }

const PbMessage& HingeLossOp::GetCustomizedConf() const { return op_conf().hinge_loss_conf(); }

LossKernelConf* HingeLossOp::GetMutLossKernelConf(KernelConf* kernel_conf) const {
  return kernel_conf->mutable_hinge_loss_conf()->mutable_loss_conf();
}

void HingeLossOp::VirtualInferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* pred_blob_desc = GetBlobDesc4BnInOp("prediction");
  BlobDesc* tmp_diff_blob_desc = GetBlobDesc4BnInOp("tmp_diff");
  tmp_diff_blob_desc->mut_shape() = Shape(pred_blob_desc->shape());
  tmp_diff_blob_desc->set_data_type(pred_blob_desc->data_type());
}

REGISTER_OP(OperatorConf::kHingeLossConf, HingeLossOp);

}  // namespace oneflow
