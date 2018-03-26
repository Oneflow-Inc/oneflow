#include "oneflow/core/operator/transpose_op.h"

namespace oneflow {

namespace {

void CheckIsPerm(const PbRf<int32_t>& perm) {
  std::vector<bool> is_used(perm.size(), 0);
  FOR_RANGE(size_t, i, 0, perm.size()) {
    CHECK_GE(perm[i], 0);
    CHECK_LT(perm[i], perm.size());
    CHECK(is_used[perm[i]] == false);
    is_used[perm[i]] = true;
  }
}

}  // namespace

void TransposeOp::InitFromOpConf() {
  CHECK(op_conf().has_transpose_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

const PbMessage& TransposeOp::GetCustomizedConf() const {
  return op_conf().transpose_conf();
}

void TransposeOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  const Shape& in_blob_shape = in_blob_desc->shape();
  const PbRf<int32_t>& perm = op_conf().transpose_conf().perm();
  // perm is for in_blob.dim[1] to in_blob.dim[n - 1]
  // example:   change blob NHWC to NCHW
  //        in_blob_shape = {100, 256, 256, 3}
  //        perm = {2, 0, 1}
  //  then: out_blob_shape = {100, 3, 256, 256}
  CHECK_EQ(perm.size(), in_blob_shape.NumAxes() - 1);
  CheckIsPerm(perm);
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  *out_blob_desc = *in_blob_desc;
  FOR_RANGE(size_t, i, 0, perm.size()) {
    out_blob_desc->mut_shape().Set(i + 1, in_blob_shape.At(perm[i] + 1));
  }
}

REGISTER_OP(OperatorConf::kTransposeConf, TransposeOp);

}  // namespace oneflow
