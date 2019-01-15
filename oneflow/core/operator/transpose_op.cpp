#include "oneflow/core/operator/transpose_op.h"

namespace oneflow {

namespace {

void CheckIsPerm(const PbRf<int32_t>& perm) {
  std::vector<bool> is_used(perm.size(), 0);
  FOR_RANGE(size_t, i, 0, perm.size()) {
    CHECK_GE(perm[i], 1);
    CHECK_LE(perm[i], perm.size());
    CHECK_EQ(is_used[perm[i] - 1], false);
    is_used[perm[i] - 1] = true;
  }
}

}  // namespace

void TransposeOp::InitFromOpConf() {
  CHECK(op_conf().has_transpose_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

const PbMessage& TransposeOp::GetCustomizedConf() const { return op_conf().transpose_conf(); }

void TransposeOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                 const ParallelContext* parallel_ctx) const {
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  const Shape& in_blob_shape = in_blob_desc->shape();
  const PbRf<int32_t>& perm = op_conf().transpose_conf().perm();
  CHECK_EQ(perm.size(), in_blob_shape.NumAxes() - 1);
  CheckIsPerm(perm);
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  *out_blob_desc = *in_blob_desc;
  FOR_RANGE(size_t, i, 0, perm.size()) {
    out_blob_desc->mut_shape().Set(i + 1, in_blob_shape.At(perm[i]));
  }
}

void TransposeOp::InferOutBlobModelSplitAxis(
    std::function<int64_t*(const std::string&)> ModelSplitAxis4BnInOp,
    std::function<int64_t(const std::string&)> ShapeNumAxes4BnInOp,
    const ParallelContext* parallel_context) const {
  int64_t in_blob_model_split_axis = *ModelSplitAxis4BnInOp("in");
  if (in_blob_model_split_axis == -1) {
    *ModelSplitAxis4BnInOp("out") = -1;
  } else {
    const PbRf<int32_t>& perm = op_conf().transpose_conf().perm();
    CHECK_GT(perm.size(), 0);
    int64_t model_split_axis = -1;
    FOR_RANGE(size_t, i, 0, perm.size()) {
      if (perm[i] == in_blob_model_split_axis) {
        model_split_axis = i;
        break;
      }
    }
    CHECK_NE(model_split_axis, -1);
    *ModelSplitAxis4BnInOp("out") = model_split_axis;
  }
}

void TransposeOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  const PbRf<int32_t>& src_perm = op_conf().transpose_conf().perm();
  PbRf<int32_t>* perm = kernel_conf->mutable_transpose_conf()->mutable_perm();
  perm->Add(0);
  perm->MergeFrom(src_perm);
  CHECK_EQ(perm->size(), src_perm.size() + 1);
  PbRf<int32_t>* invert_perm = kernel_conf->mutable_transpose_conf()->mutable_invert_perm();
  invert_perm->Reserve(perm->size());
  invert_perm->CopyFrom(*perm);
  FOR_RANGE(size_t, i, 0, perm->size()) { (*invert_perm)[(*perm)[i]] = i; }
}

REGISTER_OP(OperatorConf::kTransposeConf, TransposeOp);

}  // namespace oneflow
