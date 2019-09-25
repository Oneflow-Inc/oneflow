#include "oneflow/core/operator/accuracy_op.h"

namespace oneflow {

void AccuracyOp::InitFromOpConf() {
  EnrollInputBn("prediction", false);
  EnrollInputBn("label", false);
  EnrollOutputBn("accuracy", false);
  if (op_conf().accuracy_conf().has_weight()) {
    EnrollInputBn("weight", false);
    EnrollTmpBn("weight_reduce_tmp");
  }
}

const PbMessage& AccuracyOp::GetCustomizedConf() const { return op_conf().accuracy_conf(); }

void AccuracyOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  AccuracyKernelConf* conf = kernel_conf->mutable_accuracy_conf();
  conf->set_prediction_type(GetBlobDesc4BnInOp("prediction")->data_type());
  conf->set_label_type(GetBlobDesc4BnInOp("label")->data_type());
}

Maybe<void> AccuracyOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  BlobDesc* pred_blob_desc = GetBlobDesc4BnInOp("prediction");
  BlobDesc* label_blob_desc = GetBlobDesc4BnInOp("label");
  CHECK_OR_RETURN(IsIntegralDataType(label_blob_desc->data_type()));
  CHECK_GE_OR_RETURN(pred_blob_desc->shape().NumAxes(), 2);
  CHECK_EQ_OR_RETURN(label_blob_desc->shape(), Shape({pred_blob_desc->shape().At(0)}));
  CHECK_EQ_OR_RETURN(pred_blob_desc->is_dynamic(), label_blob_desc->is_dynamic());
  if (op_conf().accuracy_conf().has_weight()) {
    const BlobDesc* weight = GetBlobDesc4BnInOp("weight");
    CHECK_EQ_OR_RETURN(weight->shape(), label_blob_desc->shape());
    CHECK_EQ_OR_RETURN(weight->data_type(), pred_blob_desc->data_type());
    CHECK_EQ_OR_RETURN(weight->is_dynamic(), label_blob_desc->is_dynamic());

    BlobDesc* weight_reduce_tmp = GetBlobDesc4BnInOp("weight_reduce_tmp");
    weight_reduce_tmp->mut_shape() = weight->shape();
    weight_reduce_tmp->set_data_type(weight->data_type());
  }

  // accuracy
  BlobDesc* accuracy_blob_desc = GetBlobDesc4BnInOp("accuracy");
  *accuracy_blob_desc = *pred_blob_desc;
  accuracy_blob_desc->mut_shape() = Shape({1});
  accuracy_blob_desc->set_data_type(pred_blob_desc->data_type());
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kAccuracyConf, AccuracyOp);

}  // namespace oneflow
