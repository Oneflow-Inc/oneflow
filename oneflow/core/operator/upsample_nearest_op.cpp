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
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  UpsampleNearestKernelConf* conf = kernel_conf->mutable_upsample_nearest_conf();
  const bool align_corners = op_conf().upsample_nearest_conf().align_corners();
  conf->set_scale_h(GetResizeScale(in_blob_desc->shape().At(2),
                                   op_conf().upsample_nearest_conf().new_h(), align_corners));
  conf->set_scale_w(GetResizeScale(in_blob_desc->shape().At(3),
                                   op_conf().upsample_nearest_conf().new_w(), align_corners));
  conf->set_align_corners(align_corners);
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
  CHECK_GE(op_conf().upsample_nearest_conf().new_h(), 0);
  CHECK_GE(op_conf().upsample_nearest_conf().new_w(), 0);
  out_blob_desc->mut_shape() =
      Shape({in_blob_desc->shape().At(0), in_blob_desc->shape().At(1),
             op_conf().upsample_nearest_conf().new_h(), op_conf().upsample_nearest_conf().new_w()});
}

REGISTER_OP(OperatorConf::kUpsampleNearestConf, UpsampleNearestOp);

}  // namespace oneflow
