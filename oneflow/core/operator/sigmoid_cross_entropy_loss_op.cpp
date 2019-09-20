#include "oneflow/core/operator/sigmoid_cross_entropy_loss_op.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

void SigmoidCrossEntropyLossOp::VirtualInitFromOpConf() {
  EnrollTmpBn("label_num");
  EnrollTmpBn("elementwise_loss");
  EnrollTmpBn("sum_buf");
  EnrollTmpBn("count");
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
  const BlobDesc* pred_blob_desc = GetBlobDesc4BnInOp("prediction");
  BlobDesc* label_num_blob_desc = GetBlobDesc4BnInOp("label_num");
  label_num_blob_desc->mut_shape() = Shape({pred_blob_desc->shape().At(0)});
  label_num_blob_desc->set_data_type(pred_blob_desc->data_type());
  *GetBlobDesc4BnInOp("elementwise_loss") = *pred_blob_desc;
  *GetBlobDesc4BnInOp("sum_buf") = *pred_blob_desc;
  *GetBlobDesc4BnInOp("count") = *pred_blob_desc;
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kSigmoidCrossEntropyLossConf, SigmoidCrossEntropyLossOp);

}  // namespace oneflow
