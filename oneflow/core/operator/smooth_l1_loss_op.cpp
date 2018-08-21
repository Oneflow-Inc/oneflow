#include "oneflow/core/operator/smooth_l1_loss_op.h"

namespace oneflow {

void SmoothL1LossOp::VirtualInitFromOpConf() {
  EnrollInputBn("inside_weights");
  EnrollInputBn("outside_weights");
}

const PbMessage& SmoothL1LossOp::GetCustomizedConf() const {
  return op_conf().smooth_l1_loss_conf();
}

LossKernelConf* SmoothL1LossOp::GetMutLossKernelConf(KernelConf* kernel_conf) const {
  return kernel_conf->mutable_smooth_l1_loss_conf()->mutable_loss_conf();
}

void SmoothL1LossOp::VirtualInferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* pred_blob_desc = GetBlobDesc4BnInOp("prediction");
  const BlobDesc* label_blob_desc = GetBlobDesc4BnInOp("label");
  const BlobDesc* inside_weights_blob_desc = GetBlobDesc4BnInOp("inside_weights");
  const BlobDesc* outside_weights_blob_desc = GetBlobDesc4BnInOp("outside_weights");

  CHECK_EQ(pred_blob_desc->shape(), label_blob_desc->shape());
  CHECK_EQ(pred_blob_desc->shape(), inside_weights_blob_desc->shape());
  CHECK_EQ(pred_blob_desc->shape(), outside_weights_blob_desc->shape());
  CHECK_GE(pred_blob_desc->shape().NumAxes(), 2);

  BlobDesc* loss_blob_desc = GetBlobDesc4BnInOp("loss");
  loss_blob_desc->mut_shape() = Shape({pred_blob_desc->shape()});
  loss_blob_desc->set_data_type(pred_blob_desc->data_type());
  loss_blob_desc->set_has_data_id_field(pred_blob_desc->has_data_id_field());
}

REGISTER_OP(OperatorConf::kSmoothL1LossConf, SmoothL1LossOp);

}  // namespace oneflow
