#include "oneflow/core/operator/loss_op.h"

namespace oneflow {

void LossOp::InitFromOpConf() {
  EnrollInputBn("prediction");
  EnrollInputBn("label", false);
  EnrollOutputBn("loss", false);
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
  conf->set_prediction_type(GetBlobDesc4BnInOp("prediction")->body().data_type());
  conf->set_label_type(GetBlobDesc4BnInOp("label")->body().data_type());
  conf->set_weight_scalar(GetValFromCustomizedConf<float>("weight_scalar"));
  conf->set_reduction(static_cast<LossReductionType>(GetEnumFromCustomizedConf("reduction")));
}

void LossOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx, size_t* buf_size,
                            std::function<void(OpContext*)>) const {
  const BlobDesc* pred_blob_desc = GetBlobDesc4BnInOp("prediction");
  const BlobDesc* label_blob_desc = GetBlobDesc4BnInOp("label");
  CHECK_EQ(pred_blob_desc->header().has_data_id_field(),
           label_blob_desc->header().has_data_id_field());
  CHECK(IsIntegralDataType(label_blob_desc->body().data_type()));
  CHECK_GE(pred_blob_desc->body().shape().NumAxes(), 2);
  CHECK_EQ(label_blob_desc->body().shape(),
           Shape({pred_blob_desc->body().shape().At(0)}));
  // loss
  BlobDesc* loss_blob_desc = GetBlobDesc4BnInOp("loss");
  loss_blob_desc->mut_body().mut_shape() = Shape({pred_blob_desc->body().shape().At(0)});
  loss_blob_desc->mut_body().set_data_type(pred_blob_desc->body().data_type());
  loss_blob_desc->mut_header().set_has_data_id_field(
      pred_blob_desc->header().has_data_id_field());

  if (!GetValFromCustomizedConf<std::string>("weight").empty()) {
    // reduction_coefficient
    BlobDesc* reduction_blob_desc = GetBlobDesc4BnInOp("reduction_coefficient");
    reduction_blob_desc->mut_body().mut_shape() = Shape({1});
    reduction_blob_desc->mut_body().set_data_type(pred_blob_desc->body().data_type());
    reduction_blob_desc->mut_header().set_has_data_id_field(
        pred_blob_desc->header().has_data_id_field());
  }
  VirtualInferBlobDescs(GetBlobDesc4BnInOp, parallel_ctx, buf_size);
}

}  // namespace oneflow
