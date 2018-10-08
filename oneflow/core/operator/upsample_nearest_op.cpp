#include "oneflow/core/operator/upsample_nearest_op.h"

namespace oneflow {

void UpsampleNearestOp::InitFromOpConf() {
  CHECK(op_conf().has_upsample_nearest_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

const PbMessage& UpsampleNearestOp::GetCustomizedConf() const {
  return op_conf().upsample_nearest_conf();
}

void UpsampleNearestOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  ResizeNearestNeighborKernelConf* conf = kernel_conf->mutable_resize_nearest_neighbor_conf();
  const int32_t scale = op_conf().upsample_nearest_conf().scale();
  conf->set_scale_h(1.f / scale);
  conf->set_scale_w(1.f / scale);
  conf->set_align_corners(false);
}

void UpsampleNearestOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("in");
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  if (op_conf().upsample_nearest_conf().data_format() != "channels_first"
      || in_blob_desc->shape().NumAxes() != 4) {
    LOG(FATAL) << "upsample_nearest only supports NCHW";
  }
  const int32_t scale = op_conf().upsample_nearest_conf().scale();
  CHECK_GT(scale, 1);
  out_blob_desc->mut_shape() =
      Shape({in_blob_desc->shape().At(0), in_blob_desc->shape().At(1),
             scale * in_blob_desc->shape().At(2), scale * in_blob_desc->shape().At(3)});
}

REGISTER_OP(OperatorConf::kUpsampleNearestConf, UpsampleNearestOp);

}  // namespace oneflow
