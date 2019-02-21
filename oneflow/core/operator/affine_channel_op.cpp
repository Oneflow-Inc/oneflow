#include "oneflow/core/operator/affine_channel_op.h"

namespace oneflow {

void AffineChannelOp::InitFromOpConf() {
  mut_op_conf()->set_trainable(false);
  EnrollInputBn("in");
  EnrollOutputBn("out");
  EnrollModelBn("scale");
  EnrollModelBn("bias");
}

const PbMessage& AffineChannelOp::GetCustomizedConf() const {
  return op_conf().affine_channel_conf();
}

void AffineChannelOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  CHECK(op_conf().has_model_load_dir());
  const auto& conf = op_conf().affine_channel_conf();
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  const DataType in_data_type = in_blob_desc->data_type();
  CHECK_EQ(in_data_type, Global<JobDesc>::Get()->DefaultDataType());
  CHECK_GT(conf.channel_axis(), 0);
  CHECK_LE(conf.channel_axis(), in_blob_desc->shape().NumAxes());
  // out
  *GetBlobDesc4BnInOp("out") = *in_blob_desc;
  // model
  BlobDesc model_blob_desc(Shape({in_blob_desc->shape().At(conf.channel_axis())}), in_data_type,
                           false, false, 1);
  *GetBlobDesc4BnInOp("scale") = model_blob_desc;
  *GetBlobDesc4BnInOp("bias") = model_blob_desc;
}

REGISTER_OP(OperatorConf::kAffineChannelConf, AffineChannelOp);

}  // namespace oneflow
