#include "oneflow/core/operator/reduce_identity_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void ReduceIdentityOp::InitFromOpConf() {
  EnrollInputBn("in", false);
  EnrollOutputBn("out", false);
}

void ReduceIdentityOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("in");
  CHECK_EQ(GetBlobDesc4BnInOp("out")->shape().elem_cnt() % parallel_ctx->parallel_num(), 0);
}

LogicalBlobId ReduceIdentityOp::ibn2lbi(const std::string& input_bn) const {
  if (Global<JobDesc>::Get()->IsTrain()) {
    return this->Operator::ibn2lbi(input_bn);
  } else {
    return GenPackedLbi();
  }
}

LogicalBlobId ReduceIdentityOp::obn2lbi(const std::string& output_bn) const {
  return this->Operator::obn2lbi(output_bn);
}

void ReduceIdentityOp::InferSbpSignature(
    SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
    const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
    std::function<const SbpInferHint&(const std::string&)> SbpInferHint4Ibn,
    const ParallelDesc& parallel_desc) const {
  for (const auto& ibn : input_bns()) {
    CHECK(SbpInferHint4Ibn(ibn).sbp_parallel().has_partial_sum_parallel());
  }
  SbpSignatureBuilder().PartialSum(input_bns()).PartialSum(output_bns()).Build(sbp_signature);
}

REGISTER_OP(OperatorConf::kReduceIdentityConf, ReduceIdentityOp);

}  // namespace oneflow
