#include "oneflow/core/operator/multinomial_logistic_loss_op.h"

namespace oneflow {

void MultinomialLogisticLossOp::InitFromOpConf() {
  CHECK(op_conf().has_multinomial_logistic_loss_conf());

  EnrollInputBn("prediction");
  EnrollInputBn("label", false);
  EnrollOutputBn("loss", false);
  EnrollDataTmpBn("loss_buffer");
}

const PbMessage& MultinomialLogisticLossOp::GetSpecialConf() const {
  return op_conf().multinomial_logistic_loss_conf();
}

void MultinomialLogisticLossOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) {
  const BlobDesc* pred_blob_desc = GetBlobDesc4BnInOp("prediction");
  const BlobDesc* label_blob_desc = GetBlobDesc4BnInOp("label");
  CHECK_EQ(pred_blob_desc->has_data_id(), label_blob_desc->has_data_id());
  CHECK(IsIntegral(label_blob_desc->data_type()));
  // loss
  BlobDesc* loss_blob_desc = GetBlobDesc4BnInOp("loss");
  loss_blob_desc->mut_shape() = Shape({1});
  loss_blob_desc->set_data_type(pred_blob_desc->data_type());
  loss_blob_desc->set_has_data_id(false);
  // loss_buffer
  BlobDesc* loss_buffer_blob_desc = GetBlobDesc4BnInOp("loss_buffer");
  loss_buffer_blob_desc->mut_shape() = Shape({1});
  loss_buffer_blob_desc->set_data_type(pred_blob_desc->data_type());
  loss_buffer_blob_desc->set_has_data_id(false);
}

REGISTER_OP(OperatorConf::kMultinomialLogisticLossConf,
            MultinomialLogisticLossOp);

}  // namespace oneflow
