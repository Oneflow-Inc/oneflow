#include "oneflow/core/operator/loss_op.h"

namespace oneflow {

void LossOp::InitFromOpConf() {
  EnrollInputBn("prediction");
  EnrollInputBn("label", false);
  EnrollOutputBn("loss", false);
  EnrollOutputBn("loss_instance_num", false);
  if (!GetValFromCustomizedConf<std::string>("weight").empty()) {
    EnrollInputBn("weight", false);
    EnrollOutputBn("reduction_coefficient", false);
  }

  VirtualInitFromOpConf();
}

void LossOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  LossKernelConf* conf = GetMutLossKernelConf(kernel_conf);
  conf->set_prediction_type(GetBlobDesc4BnInOp("prediction")->data_type());
  conf->set_label_type(GetBlobDesc4BnInOp("label")->data_type());
  conf->set_weight_scalar(GetValFromCustomizedConf<float>("weight_scalar"));
  conf->set_reduction(static_cast<LossReductionType>(GetEnumFromCustomizedConf("reduction")));
}

void LossOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx,
                            std::function<void(OpContext*)>) const {
  const BlobDesc* pred_blob_desc = GetBlobDesc4BnInOp("prediction");
  const BlobDesc* label_blob_desc = GetBlobDesc4BnInOp("label");
  CHECK_EQ(pred_blob_desc->has_data_id_field(), label_blob_desc->has_data_id_field());
  CHECK(IsIntegralDataType(label_blob_desc->data_type()));
  CHECK_GE(pred_blob_desc->shape().NumAxes(), 2);
  CHECK_EQ(label_blob_desc->shape(), Shape({pred_blob_desc->shape().At(0)}));
  // loss
  BlobDesc* loss_blob_desc = GetBlobDesc4BnInOp("loss");
  loss_blob_desc->mut_shape() = Shape({pred_blob_desc->shape().At(0)});
  loss_blob_desc->set_data_type(pred_blob_desc->data_type());
  loss_blob_desc->set_has_data_id_field(pred_blob_desc->has_data_id_field());
  // batch instance num
  BlobDesc* loss_instance_num_blob_desc = GetBlobDesc4BnInOp("loss_instance_num");
  loss_instance_num_blob_desc->mut_shape() = Shape({1});
  loss_instance_num_blob_desc->set_data_type(pred_blob_desc->data_type());
  loss_instance_num_blob_desc->set_has_data_id_field(pred_blob_desc->has_data_id_field());

  if (!GetValFromCustomizedConf<std::string>("weight").empty()) {
    // reduction_coefficient
    BlobDesc* reduction_blob_desc = GetBlobDesc4BnInOp("reduction_coefficient");
    reduction_blob_desc->mut_shape() = Shape({1});
    reduction_blob_desc->set_data_type(pred_blob_desc->data_type());
  }
  VirtualInferBlobDescs(GetBlobDesc4BnInOp, parallel_ctx);
}

}  // namespace oneflow
