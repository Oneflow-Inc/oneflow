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
    const ParallelContext* parallel_ctx, std::function<void(OpContext*)> EnrollOpCtx) const {
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
  CHECK_GT(label_blob_desc->shape().NumAxes(), 0);
  CHECK_EQ(pred_blob_desc->shape().NumAxes(), label_blob_desc->shape().NumAxes() + 1);
  FOR_RANGE(int64_t, i, 0, label_blob_desc->shape().NumAxes()) {
    CHECK_EQ(pred_blob_desc->shape().At(i), label_blob_desc->shape().At(i));
  }
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  *out_blob_desc = *pred_blob_desc;
  out_blob_desc->mut_shape() = label_blob_desc->shape();
}

REGISTER_OP(OperatorConf::kSparseCrossEntropyConf, SparseCrossEntropyOp);

}  // namespace oneflow
