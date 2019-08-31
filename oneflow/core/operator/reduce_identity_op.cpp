#include "oneflow/core/operator/reduce_identity_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void ReduceIdentityOp::InitFromOpConf() {
  EnrollInputBn("in", false);
  EnrollOutputBn("out", false);
}

Maybe<void> ReduceIdentityOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("in");
  CHECK_EQ_OR_RETURN(GetBlobDesc4BnInOp("out")->shape().elem_cnt() % parallel_ctx->parallel_num(),
                     0);
  return Maybe<void>::Ok();
}

Maybe<void> ReduceIdentityOp::InferSbpSignature(
    SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
    const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
    std::function<const SbpInferHint&(const std::string&)> SbpInferHint4Ibn,
    const ParallelDesc& parallel_desc) const {
  for (const auto& ibn : input_bns()) {
    CHECK_OR_RETURN(SbpInferHint4Ibn(ibn).sbp_parallel().has_partial_sum_parallel());
  }
  SbpSignatureBuilder().PartialSum(input_bns()).PartialSum(output_bns()).Build(sbp_signature);
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kReduceIdentityConf, ReduceIdentityOp);

}  // namespace oneflow
