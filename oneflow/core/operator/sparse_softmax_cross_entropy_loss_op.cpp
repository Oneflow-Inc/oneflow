#include "oneflow/core/operator/sparse_softmax_cross_entropy_loss_op.h"

namespace oneflow {

void SparseSoftmaxCrossEntropyLossOp::VirtualInitFromOpConf() { EnrollDataTmpBn("prob"); }

const PbMessage& SparseSoftmaxCrossEntropyLossOp::GetCustomizedConf() const {
  return op_conf().sparse_softmax_cross_entropy_loss_conf();
}

LossKernelConf* SparseSoftmaxCrossEntropyLossOp::GetMutLossKernelConf(
    KernelConf* kernel_conf) const {
  return kernel_conf->mutable_sparse_softmax_cross_entropy_loss_conf()->mutable_loss_conf();
}

void SparseSoftmaxCrossEntropyLossOp::VirtualInferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, size_t* buf_size) const {
  const BlobDesc* pred_blob_desc = GetBlobDesc4BnInOp("prediction");
  CHECK_EQ(pred_blob_desc->shape().NumAxes(), 2);
  // prob
  BlobDesc* prob_blob_desc = GetBlobDesc4BnInOp("prob");
  prob_blob_desc->mut_shape() = Shape(pred_blob_desc->shape());
  prob_blob_desc->set_data_type(pred_blob_desc->data_type());
  *buf_size = pred_blob_desc->ByteSizeOfDataContentField();
}

REGISTER_OP(OperatorConf::kSparseSoftmaxCrossEntropyLossConf, SparseSoftmaxCrossEntropyLossOp);

}  // namespace oneflow
