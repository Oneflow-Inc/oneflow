#include "oneflow/core/operator/sigmoid_cross_entropy_loss_op.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

void SigmoidCrossEntropyLossOp::VirtualInitFromOpConf() {
  EnrollTmpBn("count");
  EnrollTmpBn("label_num");
  EnrollTmpBn("loss_buf");
  EnrollTmpBn("sum_buf");
}

const PbMessage& SigmoidCrossEntropyLossOp::GetCustomizedConf() const {
  return op_conf().sigmoid_cross_entropy_loss_conf();
}

LossKernelConf* SigmoidCrossEntropyLossOp::GetMutLossKernelConf(KernelConf* kernel_conf) const {
  return kernel_conf->mutable_sigmoid_cross_entropy_loss_conf()->mutable_loss_conf();
}

Maybe<void> SigmoidCrossEntropyLossOp::VirtualInferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  // label
  const BlobDesc* label_blob_desc = GetBlobDesc4BnInOp("label");
  // a label must be in {-1, 0, 1} while -1 indicates ignorance
  CHECK_GE_OR_RETURN(label_blob_desc->shape().NumAxes(), 2);
  // prediction
  const BlobDesc* pred_blob_desc = GetBlobDesc4BnInOp("prediction");
  CHECK_EQ_OR_RETURN(pred_blob_desc->shape(), label_blob_desc->shape());
  CHECK_GE_OR_RETURN(pred_blob_desc->shape().NumAxes(), 2);

  int64_t data_num = pred_blob_desc->shape().At(0);
  int64_t data_dim = pred_blob_desc->shape().Count(1);

  // loss
  BlobDesc* loss_blob_desc = GetBlobDesc4BnInOp("loss");
  loss_blob_desc->mut_shape() = Shape({data_num});
  loss_blob_desc->set_data_type(pred_blob_desc->data_type());
  // count
  BlobDesc* count_blob_desc = GetBlobDesc4BnInOp("count");
  count_blob_desc->mut_shape() = Shape({data_dim});
  count_blob_desc->set_data_type(pred_blob_desc->data_type());
  // loss_buf
  BlobDesc* loss_buf_desc = GetBlobDesc4BnInOp("loss_buf");
  loss_buf_desc->mut_shape() = Shape({data_dim});
  loss_buf_desc->set_data_type(pred_blob_desc->data_type());
  // label_num
  BlobDesc* label_num_blob_desc = GetBlobDesc4BnInOp("label_num");
  label_num_blob_desc->mut_shape() = Shape({1});
  label_num_blob_desc->set_data_type(pred_blob_desc->data_type());
  // sum buf
  BlobDesc* sum_buf_blob_desc = GetBlobDesc4BnInOp("sum_buf");
  const int64_t sum_buf_size = GetTmpSizeForReduceSum(pred_blob_desc->data_type(), data_dim);
  sum_buf_blob_desc->mut_shape() = Shape({sum_buf_size});
  sum_buf_blob_desc->set_data_type(DataType::kChar);
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kSigmoidCrossEntropyLossConf, SigmoidCrossEntropyLossOp);

}  // namespace oneflow
