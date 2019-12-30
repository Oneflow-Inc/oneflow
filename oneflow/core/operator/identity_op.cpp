#include "oneflow/core/operator/identity_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void IdentityOp::InitFromOpConf() {
  EnrollInputBn("in");
  EnrollOutputBn("out")->set_const_inplace_ibn("in");
}

Maybe<void> IdentityOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("in");
  return Maybe<void>::Ok();
}

const PbMessage& IdentityOp::GetCustomizedConf() const { return op_conf().identity_conf(); }

Maybe<void> IdentityOp::InferBatchAxis(
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  return NaiveInferBatchAxis(BatchAxis4BnInOp);
}

Maybe<void> IdentityOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  const auto bns = StdVec2PbRpf<std::string>({"in", "out"});
  SbpSignatureBuilder().PartialSum(bns).Build(sbp_sig_list->mutable_sbp_signature()->Add());
  const int64_t num_axes = JUST(LogicalBlobDesc4Ibn("in"))->shape().NumAxes();
  SbpSignatureBuilder().Split(bns, 0).MakeSplitSignatureListBuilder(num_axes).Build(sbp_sig_list);
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kIdentityConf, IdentityOp);

}  // namespace oneflow
