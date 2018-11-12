#include "oneflow/core/operator/broadcast_binary_op.h"

namespace oneflow {

void BroadcastBinaryOp::InitFromOpConf() {
  CHECK(op_conf().has_broadcast_add_conf());
  EnrollInputBn("a");
  EnrollInputBn("b");
  EnrollOutputBn("out");
}

void BroadcastBinaryOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* a_blob_desc = GetBlobDesc4BnInOp("a");
  const BlobDesc* b_blob_desc = GetBlobDesc4BnInOp("b");
  if (a_blob_desc->shape().elem_cnt() == 1) {
    CHECK_EQ(a_blob_desc->shape().NumAxes(), 1);
    *GetBlobDesc4BnInOp("out") = *b_blob_desc;
  } else if (b_blob_desc->shape().elem_cnt() == 1) {
    CHECK_EQ(b_blob_desc->shape().NumAxes(), 1);
    *GetBlobDesc4BnInOp("out") = *a_blob_desc;
  } else {
    CHECK_EQ(a_blob_desc->shape(), b_blob_desc->shape());
    *GetBlobDesc4BnInOp("out") = *a_blob_desc;
  }
}

}  // namespace oneflow
