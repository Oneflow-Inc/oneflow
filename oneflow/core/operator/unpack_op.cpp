#include "oneflow/core/operator/unpack_op.h"

namespace oneflow {

void UnpackOp::InitFromOpConf() {
  CHECK(op_conf().has_unpack_conf());

  EnrollInputBn("in");
  EnrollOutputBn("out");
}

void UnpackOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                              const ParallelContext* parallel_ctx) const {
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp(SoleIbn());
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp(SoleObn());
  *out_blob_desc = *in_blob_desc;
  int32_t unpack_num = op_conf().unpack_conf().unpack_num();
  if (in_blob_desc->has_dim0_inner_shape()) {
    CHECK_EQ(0, in_blob_desc->dim0_inner_shape().Count(1) % unpack_num);
  } else {
    CHECK_EQ(0, in_blob_desc->shape().At(0) % unpack_num);
  }
  out_blob_desc->mut_shape().Set(0, out_blob_desc->shape().At(0) / unpack_num);
  out_blob_desc->mut_dim0_inner_shape() = Shape({1, out_blob_desc->shape().At(0)});
}

REGISTER_OP(OperatorConf::kUnpackConf, UnpackOp);

}  // namespace oneflow
