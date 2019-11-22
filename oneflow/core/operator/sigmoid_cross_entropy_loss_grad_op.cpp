#include "oneflow/core/operator/sigmoid_cross_entropy_loss_grad_op.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

void SigmoidCrossEntropyLossGradOp::InitFromOpConf() {
  CHECK(op_conf().has_sigmoid_cross_entropy_loss_grad_conf());
  EnrollInputBn("loss_diff", false);
  EnrollInputBn("prediction", false);
  EnrollInputBn("label", false);
  EnrollOutputBn("prediction_diff");
}

const PbMessage& SigmoidCrossEntropyLossGradOp::GetCustomizedConf() const {
  return op_conf().sigmoid_cross_entropy_loss_grad_conf();
}

LossKernelConf* SigmoidCrossEntropyLossGradOp::GetMutLossKernelConf(KernelConf* kernel_conf) const {
  return kernel_conf->mutable_sigmoid_cross_entropy_loss_grad_conf()->mutable_loss_conf();
}

void SigmoidCrossEntropyLossGradOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  LossKernelConf* conf = GetMutLossKernelConf(kernel_conf);
  conf->set_prediction_type(GetBlobDesc4BnInOp("prediction")->data_type());

  if (HasFieldInCustomizedConf("label")) {
    conf->set_label_type(GetBlobDesc4BnInOp("label")->data_type());
  } else {
    conf->set_label_type(DataType::kInvalidDataType);
  }
  conf->set_weight_scalar(GetValFromCustomizedConf<float>("weight_scalar"));
  conf->set_reduction(static_cast<ScalarReductionType>(GetEnumFromCustomizedConf("reduction")));
}

Maybe<void> SigmoidCrossEntropyLossGradOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* pred_blob_desc = GetBlobDesc4BnInOp("prediction");
  CHECK_GE_OR_RETURN(pred_blob_desc->shape().NumAxes(), 2);

  const BlobDesc* label_blob_desc = GetBlobDesc4BnInOp("label");
  // a label must be in {-1, 0, 1} while -1 indicates ignorance
  CHECK_GE_OR_RETURN(label_blob_desc->shape().NumAxes(), 2);
  CHECK_EQ_OR_RETURN(pred_blob_desc->shape(), label_blob_desc->shape());

  BlobDesc* loss_diff_blob_desc = GetBlobDesc4BnInOp("prediction_diff");
  *loss_diff_blob_desc = *pred_blob_desc;
  return Maybe<void>::Ok();
}

Maybe<void> SigmoidCrossEntropyLossGradOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split(input_bns(), 0)
      .Split(output_bns(), 0)
      .MakeSplitSignatureListBuilder(JUST(LogicalBlobDesc4Ibn("prediction"))->shape().NumAxes())
      .Build(sbp_sig_list);
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kSigmoidCrossEntropyLossGradConf, SigmoidCrossEntropyLossGradOp);

}  // namespace oneflow
