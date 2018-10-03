#include "oneflow/core/operator/unpack_op.h"

namespace oneflow {

void UnpackOp::InitFromOpConf() {
  CHECK(op_conf().has_unpack_conf());

  EnrollInputBn("in", false);
  EnrollInputBn("out", false);
}

void UnpackOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                              const ParallelContext* parallel_ctx) const {
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp(SoleIbn());
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp(SoleObn());
  *out_blob_desc = *in_blob_desc;
  std::vector<int64_t> dim_vec(in_blob_desc->shape().dim_vec());
  dim_vec.at(0) = op_conf().unpack_conf().out_size();
  out_blob_desc->mut_shape() = Shape(dim_vec);
}

REGISTER_OP(OperatorConf::kUnpackConf, UnpackOp);

}  // namespace oneflow
