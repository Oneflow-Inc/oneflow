#include "oneflow/core/operator/return_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"
#include "oneflow/core/operator/interface_op_util.h"

namespace oneflow {

void ReturnOp::InitFromOpConf() {
  CHECK(op_conf().has_return_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

Maybe<void> ReturnOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("in");
  return Maybe<void>::Ok();
}

const PbMessage& ReturnOp::GetCustomizedConf() const { return op_conf().return_conf(); }

Maybe<void> ReturnOp::InferHasBatchDim(
    std::function<bool*(const std::string&)> HasBatchDim4BnInOp) const {
  *HasBatchDim4BnInOp("out") = *HasBatchDim4BnInOp("in");
  return Maybe<void>::Ok();
}

Maybe<void> ReturnOp::InferSbpSignature(
    SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
    const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
    std::function<const SbpInferHint&(const std::string&)> SbpInferHint4Ibn,
    const ParallelDesc& parallel_desc) const {
  const auto& in_sbp_infer_hint = SbpInferHint4Ibn("in");
  CHECK_OR_RETURN(in_sbp_infer_hint.parallel_desc() == parallel_desc);
  if (in_sbp_infer_hint.sbp_parallel().has_partial_sum_parallel()) {
    SbpSignatureBuilder().Broadcast(input_bns()).Broadcast(output_bns()).Build(sbp_signature);
  } else {
    auto* bn2sbp = sbp_signature->mutable_bn_in_op2sbp_parallel();
    (*bn2sbp)["in"] = in_sbp_infer_hint.sbp_parallel();
    (*bn2sbp)["out"] = in_sbp_infer_hint.sbp_parallel();
  }
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kReturnConf, ReturnOp);
REGISTER_OP_SAME_OUTPUT_BLOB_MEM_BLOCK_NUM(OperatorConf::kReturnConf, 1);
REGISTER_INTERFACE_OP(OperatorConf::kReturnConf);

}  // namespace oneflow
