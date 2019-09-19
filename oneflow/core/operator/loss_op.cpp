#include "oneflow/core/operator/loss_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void LossOp::InitFromOpConf() {
  EnrollInputBn("prediction");
  if (HasFieldInCustomizedConf("label")) { EnrollInputBn("label", false); }
  EnrollOutputBn("loss");
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
  if (HasFieldInCustomizedConf("label")) {
    conf->set_label_type(GetBlobDesc4BnInOp("label")->data_type());
  } else {
    conf->set_label_type(DataType::kInvalidDataType);
  }
  conf->set_weight_scalar(GetValFromCustomizedConf<float>("weight_scalar"));
  conf->set_reduction(static_cast<ScalarReductionType>(GetEnumFromCustomizedConf("reduction")));
}

Maybe<void> LossOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                   const ParallelContext* parallel_ctx) const {
  const BlobDesc* pred_blob_desc = GetBlobDesc4BnInOp("prediction");
  if (HasFieldInCustomizedConf("label")) {
    const BlobDesc* label_blob_desc = GetBlobDesc4BnInOp("label");
    CHECK_EQ_OR_RETURN(pred_blob_desc->has_data_id_field(), label_blob_desc->has_data_id_field());
    CHECK_EQ_OR_RETURN(pred_blob_desc->has_dim0_valid_num_field(),
                       label_blob_desc->has_dim0_valid_num_field());
    CHECK_EQ_OR_RETURN(pred_blob_desc->has_dim0_inner_shape(),
                       label_blob_desc->has_dim0_inner_shape());
  }
  if (pred_blob_desc->has_dim0_inner_shape()) {
    CHECK_EQ_OR_RETURN(pred_blob_desc->dim0_inner_shape().At(0), 1);
  }
  CHECK_GT_OR_RETURN(pred_blob_desc->shape().NumAxes(), 0);
  // loss
  BlobDesc* loss_blob_desc = GetBlobDesc4BnInOp("loss");
  *loss_blob_desc = *pred_blob_desc;
  loss_blob_desc->mut_shape() = Shape({pred_blob_desc->shape().At(0)});
  loss_blob_desc->set_data_type(pred_blob_desc->data_type());
  // loss instance num
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
  return VirtualInferBlobDescs(GetBlobDesc4BnInOp, parallel_ctx);
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

Maybe<void> LossOp::InferBatchAxis(
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  for (const auto& obn : output_bns()) { BatchAxis4BnInOp(obn)->clear_value(); }
  *BatchAxis4BnInOp("loss") = *BatchAxis4BnInOp("prediction");
  return Maybe<void>::Ok();
}

Maybe<void> LossOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split(input_bns(), 0)
      .Split(output_bns(), 0)
      .PartialSum({"loss_instance_num"})
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

}  // namespace oneflow
