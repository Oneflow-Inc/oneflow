#include "oneflow/core/operator/broadcast_like_op.h"

namespace oneflow {

void BroadcastLikeOp::InitFromOpConf() {
  EnrollInputBn("x");
  EnrollInputBn("like")->set_use_header_only(true);
  EnrollOutputBn("y");
}

const PbMessage& BroadcastLikeOp::GetCustomizedConf() const {
  return op_conf().broadcast_like_conf();
}

void BroadcastLikeOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* x_blob_desc = GetBlobDesc4BnInOp("x");
  const BlobDesc* like_blob_desc = GetBlobDesc4BnInOp("like");
  int64_t y_num_axes = std::max(like_blob_desc->shape().NumAxes(), x_blob_desc->shape().NumAxes());
  const auto& x_shape = op_conf().broadcast_like_conf().has_kept_dims_shape()
                            ? Shape(op_conf().broadcast_like_conf().kept_dims_shape())
                            : x_blob_desc->shape().CreateLeftExtendedShape(y_num_axes);
  const auto& like_shape = like_blob_desc->shape().CreateLeftExtendedShape(y_num_axes);
  FOR_RANGE(int64_t, i, 0, x_shape.NumAxes()) {
    CHECK(x_shape.At(i) == 1 || like_shape.At(i) == 1 || x_shape.At(i) == like_shape.At(i));
  }
  *GetBlobDesc4BnInOp("y") = *like_blob_desc;
}

REGISTER_OP(OperatorConf::kBroadcastLikeConf, BroadcastLikeOp);

}  // namespace oneflow
