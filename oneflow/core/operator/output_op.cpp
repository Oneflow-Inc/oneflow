#include "oneflow/core/operator/output_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"
#include "oneflow/core/operator/interface_op_util.h"

namespace oneflow {

void OutputOp::InitFromOpConf() {
  CHECK(op_conf().has_output_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

Maybe<void> OutputOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  InterfaceOpUtil::InferOutBlobDesc(op_conf().output_conf().blob_conf(), GetBlobDesc4BnInOp("out"),
                                    parallel_ctx);
  CHECK_OR_RETURN(*GetBlobDesc4BnInOp("out") == *GetBlobDesc4BnInOp("in"));
  return Maybe<void>::Ok();
}

const PbMessage& OutputOp::GetCustomizedConf() const { return op_conf().output_conf(); }

Maybe<void> OutputOp::InferHasBatchDim(
    std::function<bool*(const std::string&)> HasBatchDim4BnInOp) const {
  InterfaceOpUtil::InferHasBatchDim(op_conf().output_conf().blob_conf(), HasBatchDim4BnInOp("out"));
  CHECK_OR_RETURN(*HasBatchDim4BnInOp("out") == *HasBatchDim4BnInOp("in"));
  return Maybe<void>::Ok();
}

Maybe<void> OutputOp::InferSbpSignature(
    SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
    const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
    std::function<const SbpInferHint&(const std::string&)> SbpInferHint4Ibn,
    const ParallelDesc& parallel_desc) const {
  InterfaceOpUtil::GetOutputLikeOpSbpSignature(op_conf().output_conf().blob_conf(), input_bns(),
                                               output_bns(), sbp_signature);
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kOutputConf, OutputOp);
REGISTER_OP_SAME_OUTPUT_BLOB_MEM_BLOCK_NUM(OperatorConf::kOutputConf, 1);
REGISTER_INTERFACE_OP(OperatorConf::kOutputConf);

}  // namespace oneflow
