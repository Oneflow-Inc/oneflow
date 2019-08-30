#include "oneflow/core/operator/sparse_softmax_cross_entropy_loss_op.h"
#include "oneflow/core/register/runtime_blob_desc.h"

namespace oneflow {

void SparseSoftmaxCrossEntropyLossOp::VirtualInitFromOpConf() {
  EnrollTmpBn("prob");
  EnrollTmpBn("fw_buf");
}

const PbMessage& SparseSoftmaxCrossEntropyLossOp::GetCustomizedConf() const {
  return op_conf().sparse_softmax_cross_entropy_loss_conf();
}

LossKernelConf* SparseSoftmaxCrossEntropyLossOp::GetMutLossKernelConf(
    KernelConf* kernel_conf) const {
  return kernel_conf->mutable_sparse_softmax_cross_entropy_loss_conf()->mutable_loss_conf();
}

Maybe<void> SparseSoftmaxCrossEntropyLossOp::VirtualInferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* pred_blob_desc = GetBlobDesc4BnInOp("prediction");
  CHECK_EQ_OR_RETURN(pred_blob_desc->shape().NumAxes(), 2);
  // prob
  BlobDesc* prob_blob_desc = GetBlobDesc4BnInOp("prob");
  prob_blob_desc->mut_shape() = Shape(pred_blob_desc->shape());
  prob_blob_desc->set_data_type(pred_blob_desc->data_type());
  // temp storage for RowMax etc.
  BlobDesc* fw_buf_blob_desc = GetBlobDesc4BnInOp("fw_buf");
  fw_buf_blob_desc->mut_shape() =
      Shape({static_cast<int64_t>(RtBlobDesc(*pred_blob_desc).ByteSizeOfDataContentField())});
  fw_buf_blob_desc->set_data_type(DataType::kChar);
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kSparseSoftmaxCrossEntropyLossConf, SparseSoftmaxCrossEntropyLossOp);

}  // namespace oneflow
