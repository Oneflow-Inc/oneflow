#include "oneflow/core/operator/sigmoid_cross_entropy_loss_op.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

void SigmoidCrossEntropyLossOp::VirtualInitFromOpConf() {
  EnrollDataTmpBn("count");
  EnrollDataTmpBn("label_num");
  EnrollDataTmpBn("loss_buf");
}

const PbMessage& SigmoidCrossEntropyLossOp::GetCustomizedConf() const {
  return op_conf().sigmoid_cross_entropy_loss_conf();
}

LossKernelConf* SigmoidCrossEntropyLossOp::GetMutLossKernelConf(KernelConf* kernel_conf) const {
  return kernel_conf->mutable_sigmoid_cross_entropy_loss_conf()->mutable_loss_conf();
}

void SigmoidCrossEntropyLossOp::VirtualInferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, size_t* buf_size) const {
  // label
  const BlobDesc* label_blob_desc = GetBlobDesc4BnInOp("label");
  // a label must be in {-1, 0, 1} while -1 indicates ignorance
  CHECK_GE(label_blob_desc->shape().NumAxes(), 2);
  // prediction
  const BlobDesc* pred_blob_desc = GetBlobDesc4BnInOp("prediction");
  CHECK_EQ(pred_blob_desc->shape().elem_cnt(), label_blob_desc->shape().elem_cnt());
  CHECK_GE(pred_blob_desc->shape().NumAxes(), 2);
  // count
  BlobDesc* count_blob_desc = GetBlobDesc4BnInOp("count");
  count_blob_desc->mut_shape() = Shape(pred_blob_desc->shape());
  count_blob_desc->set_data_type(pred_blob_desc->data_type());
  // label_num
  BlobDesc* normalize_blob_desc = GetBlobDesc4BnInOp("label_num");
  normalize_blob_desc->mut_shape() = Shape({1});
  normalize_blob_desc->set_data_type(pred_blob_desc->data_type());
  // loss_buf
  BlobDesc* loss_buf_desc = GetBlobDesc4BnInOp("loss_buf");
  loss_buf_desc->mut_shape() = Shape(pred_blob_desc->shape());
  loss_buf_desc->set_data_type(pred_blob_desc->data_type());
  // loss
  BlobDesc* loss_blob_desc = GetBlobDesc4BnInOp("loss");
  loss_blob_desc->mut_shape() = Shape({1});
  loss_blob_desc->set_data_type(pred_blob_desc->data_type());
  *buf_size =
      GetTmpSizeForReduceSum(pred_blob_desc->data_type(), pred_blob_desc->shape().elem_cnt());
}

REGISTER_OP(OperatorConf::kSigmoidCrossEntropyLossConf, SigmoidCrossEntropyLossOp);

}  // namespace oneflow
