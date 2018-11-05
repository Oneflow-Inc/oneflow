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
  CHECK_EQ(pred_blob_desc->HasField<FieldKey::kDataId>(),
           label_blob_desc->HasField<FieldKey::kDataId>());
  CHECK_GE(pred_blob_desc->shape().NumAxes(), 2);
  // loss
  BlobDesc* loss_blob_desc = GetBlobDesc4BnInOp("loss");
  loss_blob_desc->mut_shape() = Shape({pred_blob_desc->shape().At(0)});
  loss_blob_desc->set_data_type(pred_blob_desc->data_type());
  loss_blob_desc->SetHasField<FieldKey::kDataId>(pred_blob_desc->HasField<FieldKey::kDataId>());
  // loss instance num
  BlobDesc* loss_instance_num_blob_desc = GetBlobDesc4BnInOp("loss_instance_num");
  loss_instance_num_blob_desc->mut_shape() = Shape({1});
  loss_instance_num_blob_desc->set_data_type(pred_blob_desc->data_type());
  loss_instance_num_blob_desc->SetHasField<FieldKey::kDataId>(
      pred_blob_desc->HasField<FieldKey::kDataId>());

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
  if (output_bn == "loss_instance_num") {
    ret.set_blob_name("loss_instance_num");
  } else if (output_bn == "reduction_coefficient") {
    ret.set_blob_name("reduction_coefficient");
  } else {
    ret.set_blob_name(GetValFromCustomizedConf<std::string>(output_bn));
  }
  return ret;
}

}  // namespace oneflow
