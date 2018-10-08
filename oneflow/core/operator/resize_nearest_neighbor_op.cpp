#include "oneflow/core/operator/resize_nearest_neighbor_op.h"

namespace oneflow {

void ResizeNearestNeighborOp::InitFromOpConf() {
  CHECK(op_conf().has_resize_nearest_neighbor_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

const PbMessage& ResizeNearestNeighborOp::GetCustomizedConf() const {
  return op_conf().resize_nearest_neighbor_conf();
}

void ResizeNearestNeighborOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  ResizeNearestNeighborKernelConf* conf = kernel_conf->mutable_resize_nearest_neighbor_conf();
  const bool align_corners = op_conf().resize_nearest_neighbor_conf().align_corners();
  conf->set_scale_h(GetResizeScale(in_blob_desc->shape().At(2),
                                   op_conf().resize_nearest_neighbor_conf().new_h(),
                                   align_corners));
  conf->set_scale_w(GetResizeScale(in_blob_desc->shape().At(3),
                                   op_conf().resize_nearest_neighbor_conf().new_w(),
                                   align_corners));
  conf->set_align_corners(op_conf().resize_nearest_neighbor_conf().align_corners());
}

void ResizeNearestNeighborOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("in");
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  if (op_conf().resize_nearest_neighbor_conf().data_format() != "channels_first"
      || in_blob_desc->shape().NumAxes() != 4) {
    LOG(FATAL) << "resize_nearest_neighbor only supports NCHW";
  }
  CHECK_GE(op_conf().resize_nearest_neighbor_conf().new_h(), 0);
  CHECK_GE(op_conf().resize_nearest_neighbor_conf().new_w(), 0);
  out_blob_desc->mut_shape() = Shape({in_blob_desc->shape().At(0), in_blob_desc->shape().At(1),
                                      op_conf().resize_nearest_neighbor_conf().new_h(),
                                      op_conf().resize_nearest_neighbor_conf().new_w()});
}

REGISTER_OP(OperatorConf::kResizeNearestNeighborConf, ResizeNearestNeighborOp);

}  // namespace oneflow
