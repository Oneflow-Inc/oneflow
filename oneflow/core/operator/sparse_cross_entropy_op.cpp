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
  CHECK_EQ(pred_blob_desc->has_dim0_inner_shape(), label_blob_desc->has_dim0_inner_shape());
  if (pred_blob_desc->has_dim0_inner_shape()) {
    CHECK_EQ(pred_blob_desc->dim0_inner_shape().At(0), 1);
    CHECK_EQ(pred_blob_desc->dim0_inner_shape(), label_blob_desc->dim0_inner_shape());
  }
  CHECK_GE(pred_blob_desc->shape().NumAxes(), 2);
  const int64_t num_out_axes = pred_blob_desc->shape().NumAxes() - 1;
  CHECK_GE(label_blob_desc->shape().NumAxes(), num_out_axes);
  CHECK_EQ(label_blob_desc->shape().Count(num_out_axes), 1);
  FOR_RANGE(int64_t, i, 0, num_out_axes) {
    CHECK_EQ(pred_blob_desc->shape().At(i), label_blob_desc->shape().At(i));
  }
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  *out_blob_desc = *pred_blob_desc;
  std::vector<int64_t> out_shape_vec = pred_blob_desc->shape().dim_vec();
  out_shape_vec.pop_back();
  out_blob_desc->mut_shape() = Shape(out_shape_vec);
}

REGISTER_OP(OperatorConf::kSparseCrossEntropyConf, SparseCrossEntropyOp);

}  // namespace oneflow
