#include "oneflow/core/operator/binary_cross_entropy_op.h"

namespace oneflow {

void BinaryCrossEntropyOp::InitFromOpConf() {
  CHECK(op_conf().has_binary_cross_entropy_conf());
  EnrollInputBn("prediction");
  EnrollInputBn("label", false);
  EnrollOutputBn("out");
}

const PbMessage& BinaryCrossEntropyOp::GetCustomizedConf() const {
  return op_conf().binary_cross_entropy_conf();
}

void BinaryCrossEntropyOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, int64_t record_piece_size,
    std::function<void(OpContext*)> EnrollOpCtx) const {
  const BlobDesc* pred_blob_desc = GetBlobDesc4BnInOp("prediction");
  const BlobDesc* label_blob_desc = GetBlobDesc4BnInOp("label");
  CHECK(IsIntegralDataType(label_blob_desc->data_type()));
  CHECK(IsFloatingDataType(pred_blob_desc->data_type()));
  CHECK_EQ(pred_blob_desc->has_data_id_field(), label_blob_desc->has_data_id_field());
  CHECK_EQ(pred_blob_desc->has_dim0_valid_num_field(), label_blob_desc->has_dim0_valid_num_field());
  CHECK_EQ(pred_blob_desc->has_dim0_inner_shape(), label_blob_desc->has_dim0_inner_shape());
  if (pred_blob_desc->has_dim0_inner_shape()) {
    CHECK_EQ(pred_blob_desc->dim0_inner_shape().At(0), 1);
    CHECK_EQ(pred_blob_desc->dim0_inner_shape(), label_blob_desc->dim0_inner_shape());
  }
  CHECK_EQ(pred_blob_desc->shape(), label_blob_desc->shape());
  *GetBlobDesc4BnInOp("out") = *pred_blob_desc;
}

REGISTER_OP(OperatorConf::kBinaryCrossEntropyConf, BinaryCrossEntropyOp);

}  // namespace oneflow
