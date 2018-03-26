#include "oneflow/core/operator/transpose_op.h"

namespace oneflow {

namespace {

void CheckIsPerm(const PbRf<int32_t>& perm) {
  std::vector<bool> is_used(perm.size(), 0);
  FOR_RANGE(size_t, i, 0, perm.size()) {
    CHECK(0 <= perm[i] && perm[i] < perm.size());
    CHECK(is_used[perm[i]] != true);
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
  CHECK_EQ(perm.size(), in_blob_shape.NumAxes());
  CHECK_EQ(perm[0], 0) << "You can't change the data num (dim[0]) of one blob";
  CheckIsPerm(perm);
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  *out_blob_desc = *in_blob_desc;
  FOR_RANGE(size_t, i, 0, perm.size()) {
    out_blob_desc->mut_shape().Set(i, in_blob_shape.At(perm[i]));
  }
}

REGISTER_OP(OperatorConf::kTransposeConf, TransposeOp);

}  // namespace oneflow
