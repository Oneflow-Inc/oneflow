#include "oneflow/core/operator/target_resize_nearest_op.h"

namespace oneflow {

void TargetResizeNearestOp::InitFromOpConf() {
  CHECK(op_conf().has_target_resize_nearest_conf());
  EnrollInputBn("in");
  EnrollInputBn("target_shape_in");
  EnrollOutputBn("out");
}

const PbMessage& TargetResizeNearestOp::GetCustomizedConf() const {
  return op_conf().target_resize_nearest_conf();
}

void TargetResizeNearestOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("in");
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  const BlobDesc* target_shape_in_blob_desc = GetBlobDesc4BnInOp("target_shape_in");
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  if (op_conf().target_resize_nearest_conf().data_format() != "channels_first"
      || in_blob_desc->shape().NumAxes() != 4) {
    LOG(FATAL) << "target_resize_nearest only supports NCHW";
  }
  CHECK_EQ(in_blob_desc->shape().At(0), target_shape_in_blob_desc->shape().At(0));
  CHECK_EQ(in_blob_desc->shape().At(1), target_shape_in_blob_desc->shape().At(1));
  out_blob_desc->mut_shape() = target_shape_in_blob_desc->shape();
}

REGISTER_OP(OperatorConf::kTargetResizeNearestConf, TargetResizeNearestOp);

}  // namespace oneflow
