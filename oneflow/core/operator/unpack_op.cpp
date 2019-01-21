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
  int32_t unpack_num = GetUnpackNum();
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

void UnpackOp::InferOutputBlobTimeShape(
    std::function<const Shape*(const std::string&)> GetTimeShape4BnInOp,
    const ParallelContext* parallel_ctx, Shape* time_shape) const {
  std::vector<int64_t> dim_vec(GetTimeShape4BnInOp("in")->dim_vec());
  int32_t unpack_num = GetUnpackNum();
  dim_vec.push_back(unpack_num);
  *time_shape = Shape(dim_vec);
}

int32_t UnpackOp::GetUnpackNum() const { return op_conf().unpack_conf().unpack_num(); }

REGISTER_OP(OperatorConf::kUnpackConf, UnpackOp);

}  // namespace oneflow
