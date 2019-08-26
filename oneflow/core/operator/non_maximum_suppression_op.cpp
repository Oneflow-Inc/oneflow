#include "oneflow/core/operator/non_maximum_suppression_op.h"

namespace oneflow {

void NonMaximumSuppressionOp::InitFromOpConf() {
  CHECK(op_conf().has_non_maximum_suppression_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
  EnrollFwBufBn("fw_tmp");
}

void NonMaximumSuppressionOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const NonMaximumSuppressionOpConf& conf = op_conf().non_maximum_suppression_conf();
  // in
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  int64_t num_boxes = in_blob_desc->shape().At(0);
  int64_t blocks =
      static_cast<int64_t>(std::ceil(num_boxes * 1.0f / GetSizeOfDataType(DataType::kInt64) * 8));
  // fw_tmp
  BlobDesc* fw_tmp_blob_desc = GetBlobDesc4BnInOp("fw_tmp");
  fw_tmp_blob_desc->mut_shape() = Shape({num_boxes * blocks});
  fw_tmp_blob_desc->set_data_type(DataType::kInt64);
  // out
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  out_blob_desc->mut_shape() = Shape({num_boxes});
  out_blob_desc->set_data_type(DataType::kInt8);
  out_blob_desc->set_has_dim0_valid_num_field(in_blob_desc->has_dim0_valid_num_field());
  out_blob_desc->mut_dim0_inner_shape() = in_blob_desc->dim0_inner_shape();
}

}  // namespace oneflow
