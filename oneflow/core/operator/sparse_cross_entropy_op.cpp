#include "oneflow/core/operator/sparse_cross_entropy_op.h"

namespace oneflow {

void SparseCrossEntropyOp::InitFromOpConf() {
  CHECK(op_conf().has_sparse_cross_entropy_conf());
  EnrollInputBn("prediction");
  EnrollInputBn("label", false);
  EnrollOutputBn("out");
}

const PbMessage& SparseCrossEntropyOp::GetCustomizedConf() const {
  return op_conf().sparse_cross_entropy_conf();
}

void SparseCrossEntropyOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, int64_t record_piece_size,
    std::function<void(OpContext*)> EnrollOpCtx) const {
  const BlobDesc* pred_blob_desc = GetBlobDesc4BnInOp("prediction");
  const BlobDesc* label_blob_desc = GetBlobDesc4BnInOp("label");
  CHECK(IsIntegralDataType(label_blob_desc->data_type()));
  CHECK(IsFloatingDataType(pred_blob_desc->data_type()));
  CHECK_EQ(pred_blob_desc->has_data_id_field(), label_blob_desc->has_data_id_field());
  CHECK_EQ(pred_blob_desc->has_dim0_valid_num_field(), label_blob_desc->has_dim0_valid_num_field());
  CHECK_GE(pred_blob_desc->shape().NumAxes(), 2);
  CHECK_EQ(label_blob_desc->shape().NumAxes(), pred_blob_desc->shape().NumAxes() - 1);

  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  *out_blob_desc = *pred_blob_desc;
  out_blob_desc->mut_shape() = Shape(std::vector<int64_t>(
      pred_blob_desc->shape().dim_vec().cbegin(), pred_blob_desc->shape().dim_vec().cend() - 1));
}

REGISTER_OP(OperatorConf::kSparseCrossEntropyConf, SparseCrossEntropyOp);

}  // namespace oneflow
