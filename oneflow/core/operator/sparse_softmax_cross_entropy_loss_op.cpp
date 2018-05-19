#include "oneflow/core/operator/sparse_softmax_cross_entropy_loss_op.h"

namespace oneflow {

void SparseSoftmaxCrossEntropyLossOp::VirtualInitFromOpConf() {
  EnrollDataTmpBn("prob");
  EnrollConstBufBn("sum_multiplier");
}

const PbMessage& SparseSoftmaxCrossEntropyLossOp::GetCustomizedConf() const {
  return op_conf().sparse_softmax_cross_entropy_loss_conf();
}

LossKernelConf* SparseSoftmaxCrossEntropyLossOp::GetMutLossKernelConf(
    KernelConf* kernel_conf) const {
  return kernel_conf->mutable_sparse_softmax_cross_entropy_loss_conf()->mutable_loss_conf();
}

void SparseSoftmaxCrossEntropyLossOp::VirtualInferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* pred_blob_desc = GetBlobDesc4BnInOp("prediction");
  CHECK_EQ(pred_blob_desc->shape().NumAxes(), 2);
  // prob
  BlobDesc* prob_blob_desc = GetBlobDesc4BnInOp("prob");
  prob_blob_desc->mut_shape() = Shape(pred_blob_desc->shape());
  prob_blob_desc->set_data_type(pred_blob_desc->data_type());

  BlobDesc* sum_multiplier_blob_desc = GetBlobDesc4BnInOp("sum_multiplier");
  sum_multiplier_blob_desc->mut_shape() = Shape({pred_blob_desc->shape().At(1)});
  sum_multiplier_blob_desc->set_data_type(pred_blob_desc->data_type());
}

REGISTER_OP(OperatorConf::kSparseSoftmaxCrossEntropyLossConf, SparseSoftmaxCrossEntropyLossOp);

}  // namespace oneflow
