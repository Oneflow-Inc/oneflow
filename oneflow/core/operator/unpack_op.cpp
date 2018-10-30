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
  int32_t unpack_num = GetUnpackNum(*parallel_ctx);
  if (in_blob_desc->has_dim0_inner_shape()) {
    CHECK_EQ(0, unpack_num % in_blob_desc->dim0_inner_shape().At(0));
    CHECK_EQ(0, in_blob_desc->dim0_inner_shape().Count(1)
                    % (unpack_num / in_blob_desc->dim0_inner_shape().At(0)));
  } else {
    CHECK_EQ(0, in_blob_desc->shape().At(0) % unpack_num);
  }
  out_blob_desc->mut_shape().Set(0, out_blob_desc->shape().At(0) / unpack_num);
  out_blob_desc->mut_dim0_inner_shape() = Shape({1, out_blob_desc->shape().At(0)});
}

int32_t UnpackOp::GetUnpackNum(const ParallelContext& ctx) const {
  CHECK(op_conf().has_unpack_conf());
  const UnpackOpConf& conf = op_conf().unpack_conf();
  if (conf.has_unpack_num()) {
    return conf.unpack_num();
  } else if (conf.has_unpack_num_per_record()) {
    int64_t unpack_num =
        Global<JobDesc>::Get()->DevicePieceSize4ParallelCtx(ctx) * conf.unpack_num_per_record();
    CHECK_LE(unpack_num, static_cast<int64_t>(MaxVal<int32_t>()));
    return static_cast<int32_t>(unpack_num);
  } else {
    UNIMPLEMENTED();
  }
}

REGISTER_OP(OperatorConf::kUnpackConf, UnpackOp);

}  // namespace oneflow
