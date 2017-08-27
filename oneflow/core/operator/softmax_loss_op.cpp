#include "oneflow/core/operator/softmax_loss_op.h"
#include "oneflow/core/common/data_type.h"

namespace oneflow {

void SoftmaxLossOp::InitFromOpConf() {
  CHECK(op_conf().has_softmax_loss_conf());
  EnrollInputBn("prediction");
  EnrollInputBn("label", false);
  EnrollDataTmpBn("prob");
  EnrollDataTmpBn("tmp_1D");
  EnrollOutputBn("loss", false);
}

const PbMessage& SoftmaxLossOp::GetSpecialConf() const {
  return op_conf().softmax_loss_conf();
}

void SoftmaxLossOp::InferBlobDesc4FwBlobs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    ParallelPolicy policy, int64_t parallel_id, int64_t parallel_num) const {
  const SoftmaxLossOpConf& conf = op_conf().softmax_loss_conf();
  // CHECK data type
  const BlobDesc* pred_blob_desc = GetBlobDesc4BnInOp("prediction");
  CHECK_EQ(pred_blob_desc->data_type(), conf.prediction().data_type());
  CHECK_EQ(conf.prediction().data_type(),
           JobDesc::Singleton()->default_data_type());
  CHECK_EQ(conf.loss().data_type(), JobDesc::Singleton()->default_data_type());
  const BlobDesc* label_blob_desc = GetBlobDesc4BnInOp("label");
  CHECK_EQ(pred_blob_desc->has_data_id(), label_blob_desc->has_data_id());
  // CHECK label data type is int32_t
  CHECK_EQ(label_blob_desc->data_type(), DataType::kInt32);
  // CHECK shape
  CHECK_EQ(pred_blob_desc->shape().NumAxes(), 2);
  CHECK_EQ(label_blob_desc->shape(), Shape({pred_blob_desc->shape().At(0)}));
  // loss
  BlobDesc* loss_blob_desc = GetBlobDesc4BnInOp("loss");
  loss_blob_desc->mut_shape() = Shape({1});
  loss_blob_desc->set_data_type(conf.loss().data_type());
  loss_blob_desc->set_has_data_id(false);
  // tmp_1D
  BlobDesc* tmp_1D_blob_desc = GetBlobDesc4BnInOp("tmp_1D");
  tmp_1D_blob_desc->mut_shape() = Shape({pred_blob_desc->shape().At(0)});
  tmp_1D_blob_desc->set_data_type(conf.loss().data_type());
  tmp_1D_blob_desc->set_has_data_id(false);
  // prob
  BlobDesc* prob_blob_desc = GetBlobDesc4BnInOp("prob");
  prob_blob_desc->mut_shape() = Shape(pred_blob_desc->shape());
  prob_blob_desc->set_data_type(conf.loss().data_type());
  prob_blob_desc->set_has_data_id(false);
}

REGISTER_OP(OperatorConf::kSoftmaxLossConf, SoftmaxLossOp);

}  // namespace oneflow
