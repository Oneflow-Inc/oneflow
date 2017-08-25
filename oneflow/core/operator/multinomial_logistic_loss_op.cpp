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

void MultinomialLogisticLossOp::InferBlobDesc4FwBlobs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    ParallelPolicy policy, int64_t parallel_id, int64_t parallel_num) const {
  const MultinomialLogisticLossOpConf& conf =
      op_conf().multinomial_logistic_loss_conf();
  const BlobDesc* pred_blob_desc = GetBlobDesc4BnInOp("prediction");
  CHECK_EQ(pred_blob_desc->data_type(), conf.prediction().data_type());
  CHECK_EQ(conf.prediction().data_type(),
           JobDesc::Singleton()->default_data_type());
  CHECK_EQ(conf.loss().data_type(), JobDesc::Singleton()->default_data_type());
  const BlobDesc* label_blob_desc = GetBlobDesc4BnInOp("label");
  CHECK_EQ(pred_blob_desc->has_data_id(), label_blob_desc->has_data_id());
  // CHECK label data type is int32_t
  CHECK_EQ(label_blob_desc->data_type(), DataType::kInt32);
  // loss
  BlobDesc* loss_blob_desc = GetBlobDesc4BnInOp("loss");
  loss_blob_desc->mut_shape() = Shape({1});
  loss_blob_desc->set_data_type(conf.loss().data_type());
  loss_blob_desc->set_has_data_id(false);
  // loss_buffer
  BlobDesc* loss_buffer_blob_desc = GetBlobDesc4BnInOp("loss");
  loss_buffer_blob_desc->mut_shape() = Shape({1});
  loss_buffer_blob_desc->set_data_type(conf.loss().data_type());
  loss_buffer_blob_desc->set_has_data_id(false);
}

REGISTER_OP(OperatorConf::kMultinomialLogisticLossConf,
            MultinomialLogisticLossOp);

}  // namespace oneflow
