#include "oneflow/core/operator/transpose_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

namespace {

void CheckIsPerm(const PbRf<int32_t>& perm) {
  std::vector<bool> is_used(perm.size(), false);
  FOR_RANGE(size_t, i, 0, perm.size()) {
    CHECK_GE(perm[i], 0);
    CHECK_LE(perm[i], perm.size());
    CHECK_EQ(is_used[perm[i]], false);
    is_used[perm[i]] = true;
  }
}

}  // namespace

void TransposeOp::InitFromOpConf() {
  CHECK(op_conf().has_transpose_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

const PbMessage& TransposeOp::GetCustomizedConf() const { return op_conf().transpose_conf(); }

Maybe<void> TransposeOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  const Shape& in_blob_shape = in_blob_desc->shape();
  const PbRf<int32_t>& perm = op_conf().transpose_conf().perm();
  CHECK_EQ_OR_RETURN(perm.size(), in_blob_shape.NumAxes());
  CheckIsPerm(perm);
  if (perm.Get(0) != 0) {
    CHECK_OR_RETURN(!in_blob_desc->has_dim0_valid_num_field());
  } else if (perm.size() >= 2 && perm.Get(1) != 1) {
    CHECK_OR_RETURN(!in_blob_desc->has_dim1_valid_num_field());
  } else if (perm.size() >= 3 && perm.Get(2) != 2) {
    CHECK_OR_RETURN(!in_blob_desc->has_dim2_valid_num_field());
  } else {
    // do nothing
  }
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  *out_blob_desc = *in_blob_desc;
  FOR_RANGE(size_t, i, 0, perm.size()) {
    out_blob_desc->mut_shape().Set(i, in_blob_shape.At(perm[i]));
  }
  return Maybe<void>::Ok();
}

void TransposeOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  const PbRf<int32_t>& src_perm = op_conf().transpose_conf().perm();
  PbRf<int32_t>* perm = kernel_conf->mutable_transpose_conf()->mutable_perm();
  *perm = src_perm;
  CHECK_EQ(perm->size(), src_perm.size());
  PbRf<int32_t>* invert_perm = kernel_conf->mutable_transpose_conf()->mutable_invert_perm();
  invert_perm->Reserve(perm->size());
  invert_perm->CopyFrom(*perm);
  FOR_RANGE(size_t, i, 0, perm->size()) { (*invert_perm)[(*perm)[i]] = i; }
}

Maybe<void> TransposeOp::InferBatchAxis(
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  if (BatchAxis4BnInOp("in")->has_value()) {
    const PbRf<int32_t>& perm = op_conf().transpose_conf().perm();
    BatchAxis4BnInOp("out")->set_value(perm.Get(BatchAxis4BnInOp("in")->value()));
  } else {
    BatchAxis4BnInOp("out")->clear_value();
  }
  return Maybe<void>::Ok();
}

Maybe<void> TransposeOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  const PbRf<int32_t>& perm = op_conf().transpose_conf().perm();
  CHECK_EQ(perm.size(), JUST(LogicalBlobDesc4Ibn("in"))->shape().NumAxes());
  FOR_RANGE(int32_t, i, 0, perm.size()) {
    int32_t axis = perm.Get(i);
    if (axis < 0) { axis += perm.size(); }
    CHECK_GE(axis, 0);
    CHECK_LT(axis, perm.size());
    SbpSignatureBuilder()
        .Split(input_bns(), i)
        .Split(output_bns(), axis)
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  }
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kTransposeConf, TransposeOp);

}  // namespace oneflow
