#include "oneflow/core/operator/gather_ms0_op.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

void GatherMs0Op::InitFromOpConf() {
  CHECK(op_conf().has_gather_ms0_conf());
  EnrollInputBn("indices", false);
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

const PbMessage& GatherMs0Op::GetCustomizedConf() const { return op_conf().gather_ms0_conf(); }

Maybe<void> GatherMs0Op::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* indices = GetBlobDesc4BnInOp("indices");
  CHECK(IsIntegralDataType(indices->data_type()));
  CHECK_GT(indices->shape().NumAxes(), 0);
  const BlobDesc* in = GetBlobDesc4BnInOp("in");
  CHECK_GT(in->shape().NumAxes(), 0);
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  *out = *indices;
  std::vector<int64_t> dim_vec = indices->shape().dim_vec();
  FOR_RANGE(int, i, 1, in->shape().NumAxes()) { dim_vec.push_back(in->shape().At(i)); }
  out->mut_shape() = Shape(dim_vec);
  return Maybe<void>::Ok();
}

Maybe<void> GatherMs0Op::GetSbpSignatures(SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder().Broadcast("indices").Split("in", 0).PartialSum("out").Build(
      sbp_sig_list->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

void GatherMs0Op::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf, const OpContext* op_ctx,
    std::function<const BlobDesc&(const std::string&)> LogicalBlobDesc4BnInOp) const {
  int64_t dim = LogicalBlobDesc4BnInOp("in").shape().At(0);
  CHECK_GE(dim, parallel_ctx->parallel_num());
  BalancedSplitter bs(dim, parallel_ctx->parallel_num());
  kernel_conf->mutable_gather_ms0_conf()->set_offset(bs.At(parallel_ctx->parallel_id()).begin());
}

REGISTER_OP(OperatorConf::kGatherMs0Conf, GatherMs0Op);

}  // namespace oneflow
