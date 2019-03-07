#include "oneflow/core/operator/affine_channel_op.h"

namespace oneflow {

void AffineChannelOp::InitFromOpConf() {
  CHECK(op_conf().has_affine_channel_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
  EnrollModelBn("scale");
  if (GetValFromCustomizedConf<bool>("use_bias")) { EnrollModelBn("bias"); }
}

void AffineChannelOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const auto& conf = op_conf().affine_channel_conf();
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  CHECK_GE(conf.axis(), -in_blob_desc->shape().NumAxes());
  CHECK_LT(conf.axis(), in_blob_desc->shape().NumAxes());
  const int32_t axis =
      conf.axis() >= 0 ? conf.axis() : conf.axis() + in_blob_desc->shape().NumAxes();

  *GetBlobDesc4BnInOp("out") = *in_blob_desc;
  BlobDesc model_blob_desc(Shape({in_blob_desc->shape().At(axis)}), in_blob_desc->data_type(),
                           false, false, 1);
  *GetBlobDesc4BnInOp("scale") = model_blob_desc;
  if (GetValFromCustomizedConf<bool>("use_bias")) { *GetBlobDesc4BnInOp("bias") = model_blob_desc; }
}

const PbMessage& AffineChannelOp::GetCustomizedConf() const {
  return op_conf().affine_channel_conf();
}

REGISTER_OP(OperatorConf::kAffineChannelConf, AffineChannelOp);

}  // namespace oneflow
