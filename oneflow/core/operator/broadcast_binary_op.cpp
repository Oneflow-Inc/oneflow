#include "oneflow/core/operator/broadcast_binary_op.h"

namespace oneflow {

namespace {

bool IsScalarBlob(const BlobDesc* blob) {
  return blob->shape().NumAxes() == 1 && blob->shape().At(0) == 1;
}

}  // namespace

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
  if (IsScalarBlob(a_blob_desc)) {
    *out_blob_desc = *b_blob_desc;
  } else if (IsScalarBlob(b_blob_desc)) {
    *out_blob_desc = *a_blob_desc;
  } else {
    const auto& a_shape = a_blob_desc->shape();
    const auto& b_shape = b_blob_desc->shape();
    CHECK_EQ(a_shape.NumAxes(), b_shape.NumAxes());
    *out_blob_desc = *a_blob_desc;
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
