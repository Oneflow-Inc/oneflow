#include "oneflow/core/operator/center_loss_op.h"
#include "oneflow/core/register/runtime_blob_desc.h"

namespace oneflow {
void CenterLossOp::VirtualInitFromOpConf() {
  EnrollForwardModelBn("centers");
  EnrollDataTmpBn("piece_centers");
  EnrollDataTmpBn("forward_tmp");
  EnrollConstBufBn("ones_multipiler");
}

const PbMessage& CenterLossOp::GetCustomizedConf() const { return op_conf().center_loss_conf(); }

LossKernelConf* CenterLossOp::GetMutLossKernelConf(KernelConf* kernel_conf) const {
  return kernel_conf->mutable_center_loss_conf()->mutable_loss_conf();
}

void CenterLossOp::VirtualInferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* prediction_blob_desc = GetBlobDesc4BnInOp("prediction");
  CHECK_EQ(prediction_blob_desc->shape().NumAxes(), 2);

  // centers [num_of_classes, dim]
  BlobDesc* centers_blob_desc = GetBlobDesc4BnInOp("centers");
  *centers_blob_desc = *prediction_blob_desc;
  centers_blob_desc->mut_shape() = Shape(
      {this->op_conf().center_loss_conf().num_of_classes(), prediction_blob_desc->shape().At(1)});

  // piece_centers [piece_size, dim]
  BlobDesc* piece_centers_blob_desc = GetBlobDesc4BnInOp("piece_centers");
  *piece_centers_blob_desc = *prediction_blob_desc;

  // forward_tmp [piece_size, dim]
  BlobDesc* forward_tmp_blob_desc = GetBlobDesc4BnInOp("forward_tmp");
  *forward_tmp_blob_desc = *prediction_blob_desc;

  // ones_multipiler [dim]
  BlobDesc* ones_multipiler_blob_desc = GetBlobDesc4BnInOp("ones_multipiler");
  *ones_multipiler_blob_desc = *prediction_blob_desc;
  ones_multipiler_blob_desc->mut_shape() = Shape({prediction_blob_desc->shape().At(1)});
}

REGISTER_OP(OperatorConf::kCenterLossConf, CenterLossOp);

}  // namespace oneflow
