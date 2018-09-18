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
  out_blob_desc->mut_shape() =
      Shape({in_blob_desc->shape().At(0), in_blob_desc->shape().At(1),
             scale * in_blob_desc->shape().At(2), scale * in_blob_desc->shape().At(3)});
}

REGISTER_OP(OperatorConf::kUpsampleNearestConf, UpsampleNearestOp);

}  // namespace oneflow
