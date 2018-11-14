#include "oneflow/core/operator/broadcast_binary_op.h"

namespace oneflow {

void BroadcastBinaryOp::InitFromOpConf() {
  EnrollInputBn("a");
  EnrollInputBn("b");
  EnrollOutputBn("out");
  EnrollBwBufBn("bw_buf");
}

void BroadcastBinaryOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* a_blob_desc = GetBlobDesc4BnInOp("a");
  const BlobDesc* b_blob_desc = GetBlobDesc4BnInOp("b");
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  if (a_blob_desc->shape().elem_cnt() == 1) {
    *out_blob_desc = *b_blob_desc;
  } else if (b_blob_desc->shape().elem_cnt() == 1) {
    *out_blob_desc = *a_blob_desc;
  } else {
    *out_blob_desc = *a_blob_desc;
    const auto& a_shape = a_blob_desc->shape();
    const auto& b_shape = b_blob_desc->shape();
    CHECK_EQ(a_shape.NumAxes(), b_shape.NumAxes());
    FOR_RANGE(int64_t, i, 0, a_shape.NumAxes()) {
      CHECK(a_shape.At(i) == 1 || b_shape.At(i) == 1 || a_shape.At(i) == b_shape.At(i));
      out_blob_desc->mut_shape().Set(i, std::max(a_shape.At(i), b_shape.At(i)));
    }
  }
}

void BroadcastBinaryOp::InferBwBufBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp, const ParallelContext*) {
  BlobDesc* bw_buf = GetBlobDesc4BnInOp("bw_buf");
  bw_buf->mut_shape() = Shape({GetBlobDesc4BnInOp("out")->shape().elem_cnt()});
  bw_buf->set_data_type(DataType::kChar);
}

}  // namespace oneflow
