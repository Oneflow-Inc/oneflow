#include "oneflow/core/operator/accuracy_op.h"

namespace oneflow {

void AccuracyOp::InitFromOpConf() {
  EnrollInputBn("prediction", false);
  EnrollInputBn("label", false);
  EnrollOutputBn("accuracy", false);
}

const PbMessage& AccuracyOp::GetCustomizedConf() const { return op_conf().accuracy_conf(); }

void AccuracyOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  AccuracyKernelConf* conf = kernel_conf->mutable_accuracy_conf();
  conf->set_prediction_type(GetBlobDesc4BnInOp("prediction")->body_desc().data_type());
  conf->set_label_type(GetBlobDesc4BnInOp("label")->body_desc().data_type());
}

void AccuracyOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx, size_t* buf_size,
                                std::function<void(OpContext*)>) const {
  BlobDesc* pred_blob_desc = GetBlobDesc4BnInOp("prediction");
  BlobDesc* label_blob_desc = GetBlobDesc4BnInOp("label");
  CHECK_EQ(pred_blob_desc->header_desc().has_data_id_field(),
           label_blob_desc->header_desc().has_data_id_field());
  CHECK(IsIntegralDataType(label_blob_desc->body_desc().data_type()));
  CHECK_GE(pred_blob_desc->body_desc().shape().NumAxes(), 2);
  CHECK_EQ(label_blob_desc->body_desc().shape(),
           Shape({pred_blob_desc->body_desc().shape().At(0)}));

  // accuracy output blob
  BlobDesc* accuracy_blob_desc = GetBlobDesc4BnInOp("accuracy");
  accuracy_blob_desc->mut_body_desc().mut_shape() = Shape({1});
  accuracy_blob_desc->mut_body_desc().set_data_type(pred_blob_desc->body_desc().data_type());
  accuracy_blob_desc->mut_header_desc().set_has_data_id_field(
      pred_blob_desc->header_desc().has_data_id_field());
}

REGISTER_OP(OperatorConf::kAccuracyConf, AccuracyOp);

}  // namespace oneflow
