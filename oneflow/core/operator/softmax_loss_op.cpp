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

void SoftmaxLossOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) {
  const BlobDesc* pred_blob_desc = GetBlobDesc4BnInOp("prediction");
  const BlobDesc* label_blob_desc = GetBlobDesc4BnInOp("label");
  CHECK_EQ(pred_blob_desc->has_data_id(), label_blob_desc->has_data_id());
  CHECK(IsIntegral(label_blob_desc->data_type()));
  CHECK_EQ(pred_blob_desc->shape().NumAxes(), 2);
  CHECK_EQ(label_blob_desc->shape(), Shape({pred_blob_desc->shape().At(0)}));
  // loss
  BlobDesc* loss_blob_desc = GetBlobDesc4BnInOp("loss");
  loss_blob_desc->mut_shape() = Shape({1});
  loss_blob_desc->set_data_type(pred_blob_desc->data_type());
  loss_blob_desc->set_has_data_id(false);
  // tmp_1D
  BlobDesc* tmp_1D_blob_desc = GetBlobDesc4BnInOp("tmp_1D");
  tmp_1D_blob_desc->mut_shape() = Shape({pred_blob_desc->shape().At(0)});
  tmp_1D_blob_desc->set_data_type(pred_blob_desc->data_type());
  tmp_1D_blob_desc->set_has_data_id(false);
  // prob
  BlobDesc* prob_blob_desc = GetBlobDesc4BnInOp("prob");
  prob_blob_desc->mut_shape() = Shape(pred_blob_desc->shape());
  prob_blob_desc->set_data_type(pred_blob_desc->data_type());
  prob_blob_desc->set_has_data_id(false);
}

REGISTER_OP(OperatorConf::kSoftmaxLossConf, SoftmaxLossOp);

}  // namespace oneflow
