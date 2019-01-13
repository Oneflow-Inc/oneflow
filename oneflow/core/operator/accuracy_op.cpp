#include "oneflow/core/operator/accuracy_op.h"

namespace oneflow {

void AccuracyOp::InitFromOpConf() {
  EnrollInputBn("prediction", false);
  EnrollInputBn("label", false);
  EnrollOutputBn("accuracy", false);
  EnrollOutputBn("accuracy_instance_num", false);
}

const PbMessage& AccuracyOp::GetCustomizedConf() const { return op_conf().accuracy_conf(); }

void AccuracyOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  AccuracyKernelConf* conf = kernel_conf->mutable_accuracy_conf();
  conf->set_prediction_type(GetBlobDesc4BnInOp("prediction")->data_type());
  conf->set_label_type(GetBlobDesc4BnInOp("label")->data_type());
}

void AccuracyOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx) const {
  BlobDesc* pred_blob_desc = GetBlobDesc4BnInOp("prediction");
  BlobDesc* label_blob_desc = GetBlobDesc4BnInOp("label");
  CHECK_EQ(pred_blob_desc->has_data_id_field(), label_blob_desc->has_data_id_field());
  CHECK_EQ(pred_blob_desc->has_dim0_valid_num_field(), label_blob_desc->has_dim0_valid_num_field());
  CHECK_EQ(pred_blob_desc->has_dim0_inner_shape(), label_blob_desc->has_dim0_inner_shape());
  if (pred_blob_desc->has_dim0_inner_shape()) {
    CHECK_EQ(pred_blob_desc->dim0_inner_shape().At(0), 1);
  }
  CHECK(IsIntegralDataType(label_blob_desc->data_type()));
  CHECK_GE(pred_blob_desc->shape().NumAxes(), 2);
  CHECK_EQ(label_blob_desc->shape(), Shape({pred_blob_desc->shape().At(0)}));

  // accuracy
  BlobDesc* accuracy_blob_desc = GetBlobDesc4BnInOp("accuracy");
  *accuracy_blob_desc = *pred_blob_desc;
  accuracy_blob_desc->mut_shape() = Shape({1});
  accuracy_blob_desc->set_data_type(pred_blob_desc->data_type());

  // accuracy instance num
  BlobDesc* accuracy_instance_num_blob_desc = GetBlobDesc4BnInOp("accuracy_instance_num");
  accuracy_instance_num_blob_desc->mut_shape() = Shape({1});
  accuracy_instance_num_blob_desc->set_data_type(pred_blob_desc->data_type());
  accuracy_instance_num_blob_desc->set_has_data_id_field(pred_blob_desc->has_data_id_field());
}

LogicalBlobId AccuracyOp::obn2lbi(const std::string& output_bn) const {
  LogicalBlobId ret;
  ret.set_op_name(op_name());
  if (output_bn == "accuracy_instance_num") {
    ret.set_blob_name("accuracy_instance_num");
  } else {
    ret.set_blob_name(GetValFromCustomizedConf<std::string>(output_bn));
  }
  return ret;
}

REGISTER_OP(OperatorConf::kAccuracyConf, AccuracyOp);

}  // namespace oneflow
