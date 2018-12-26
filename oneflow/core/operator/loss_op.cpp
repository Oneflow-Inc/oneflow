#include "oneflow/core/operator/loss_op.h"

namespace oneflow {

void LossOp::InitFromOpConf() {
  EnrollInputBn("prediction");
  if (HasFieldInCustomizedConf("label")) { EnrollInputBn("label", false); }
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
  conf->set_prediction_type(GetBlobDesc4BnInOp("prediction")->data_type());
  if (HasFieldInCustomizedConf("label")) {
    conf->set_label_type(GetBlobDesc4BnInOp("label")->data_type());
  } else {
    conf->set_label_type(DataType::kInvalidDataType);
  }
  conf->set_weight_scalar(GetValFromCustomizedConf<float>("weight_scalar"));
  conf->set_reduction(static_cast<LossReductionType>(GetEnumFromCustomizedConf("reduction")));
}

void LossOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx,
                            std::function<void(OpContext*)>) const {
  const BlobDesc* pred_blob_desc = GetBlobDesc4BnInOp("prediction");
  if (HasFieldInCustomizedConf("label")) {
    const BlobDesc* label_blob_desc = GetBlobDesc4BnInOp("label");
    CHECK_EQ(pred_blob_desc->has_data_id_field(), label_blob_desc->has_data_id_field());
    CHECK(IsIntegralDataType(label_blob_desc->data_type()));
    CHECK_GE(pred_blob_desc->shape().NumAxes(), 2);
    CHECK_EQ(label_blob_desc->shape(), Shape({pred_blob_desc->shape().At(0)}));
  }
  CHECK_GT(pred_blob_desc->shape().NumAxes(), 0);
  // loss
  BlobDesc* loss_blob_desc = GetBlobDesc4BnInOp("loss");
  *loss_blob_desc = *pred_blob_desc;
  loss_blob_desc->mut_shape() = Shape({pred_blob_desc->shape().At(0)});
  loss_blob_desc->set_data_type(pred_blob_desc->data_type());

  if (!GetValFromCustomizedConf<std::string>("weight").empty()) {
    // reduction_coefficient
    BlobDesc* reduction_blob_desc = GetBlobDesc4BnInOp("reduction_coefficient");
    reduction_blob_desc->mut_shape() = Shape({1});
    reduction_blob_desc->set_data_type(pred_blob_desc->data_type());
  }
  VirtualInferBlobDescs(GetBlobDesc4BnInOp, parallel_ctx);
}

LogicalBlobId LossOp::obn2lbi(const std::string& output_bn) const {
  LogicalBlobId ret;
  ret.set_op_name(op_name());
  if (output_bn == "reduction_coefficient") {
    ret.set_blob_name("reduction_coefficient");
  } else {
    ret.set_blob_name(GetValFromCustomizedConf<std::string>(output_bn));
  }
  return ret;
}

}  // namespace oneflow
