#include "oneflow/core/operator/sigmoid_cross_entropy_loss_grad_op.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

void SigmoidCrossEntropyLossGradOp::InitFromOpConf() {
  CHECK(op_conf().has_sigmoid_cross_entropy_loss_grad_conf());
  EnrollInputBn("loss_diff");
  EnrollOutputBn("prediction");
  EnrollOutputBn("label");
}

const PbMessage& SigmoidCrossEntropyLossGradOp::GetCustomizedConf() const {
  return op_conf().sigmoid_cross_entropy_loss_grad_conf();
}

void SigmoidCrossEntropyLossGradOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  // prediction
  const BlobDesc* pred_blob_desc = GetBlobDesc4BnInOp("prediction");
  CHECK_GE(pred_blob_desc->shape().NumAxes(), 2);
  int64_t data_num = pred_blob_desc->shape().At(0);

  // label
  const BlobDesc* label_blob_desc = GetBlobDesc4BnInOp("label");
  // a label must be in {-1, 0, 1} while -1 indicates ignorance
  CHECK_GE(label_blob_desc->shape().NumAxes(), 2);
  CHECK_EQ(pred_blob_desc->shape(), label_blob_desc->shape());

  // loss
  BlobDesc* loss_diff_blob_desc = GetBlobDesc4BnInOp("loss_diff");
  loss_diff_blob_desc->mut_shape() = Shape({data_num});
  loss_diff_blob_desc->set_data_type(pred_blob_desc->data_type());
   
}

void SigmoidCrossEntropyLossGradOp::GetSbpSignatures(
     const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
     SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split(input_bns(), 0)
      .Split(output_bns(), 0)
      .MakeSplitSignatureListBuilder(LogicalBlobDesc4Ibn("in").shape().NumAxes())
      .Build(sbp_sig_list);
}


REGISTER_OP(OperatorConf::kSigmoidCrossEntropyLossGradConf, SigmoidCrossEntropyLossGradOp);

}  // namespace oneflow
