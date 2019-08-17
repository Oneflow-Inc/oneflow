#include "oneflow/core/operator/output_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"
#include "oneflow/core/operator/interface_op_util.h"

namespace oneflow {

void OutputOp::InitFromOpConf() {
  CHECK(op_conf().has_output_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

void OutputOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                              const ParallelContext* parallel_ctx) const {
  if (op_conf().output_conf().has_blob_conf()) {
    InterfaceOpUtil::InferOutBlobDesc(op_conf().output_conf().blob_conf(),
                                      GetBlobDesc4BnInOp("out"), parallel_ctx);
    CHECK(*GetBlobDesc4BnInOp("out") == *GetBlobDesc4BnInOp("in"));
  } else {
    *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("in");
  }
}

const PbMessage& OutputOp::GetCustomizedConf() const { return op_conf().output_conf(); }

void OutputOp::InferHasBatchDim(std::function<bool*(const std::string&)> HasBatchDim4BnInOp) const {
  if (op_conf().output_conf().has_blob_conf()) {
    InterfaceOpUtil::InferHasBatchDim(op_conf().output_conf().blob_conf(),
                                      HasBatchDim4BnInOp("out"));
    CHECK(*HasBatchDim4BnInOp("out") == *HasBatchDim4BnInOp("in"));
  } else {
    *HasBatchDim4BnInOp("out") = *HasBatchDim4BnInOp("in");
  }
}

void OutputOp::InferSbpSignature(
    SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
    const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
    std::function<const SbpInferHint&(const std::string&)> SbpInferHint4Ibn,
    const ParallelDesc& parallel_desc) const {
  if (op_conf().output_conf().has_blob_conf()) {
    InterfaceOpUtil::GetOutputLikeOpSbpSignature(op_conf().output_conf().blob_conf(), input_bns(),
                                                 output_bns(), sbp_signature);
  } else {
    const auto& in_sbp_infer_hint = SbpInferHint4Ibn("in");
    CHECK(in_sbp_infer_hint.parallel_desc() == parallel_desc);
    if (in_sbp_infer_hint.sbp_parallel().has_partial_sum_parallel()) {
      SbpSignatureBuilder().Broadcast(input_bns()).Broadcast(output_bns()).Build(sbp_signature);
    } else {
      auto* bn2sbp = sbp_signature->mutable_bn_in_op2sbp_parallel();
      (*bn2sbp)["in"] = in_sbp_infer_hint.sbp_parallel();
      (*bn2sbp)["out"] = in_sbp_infer_hint.sbp_parallel();
    }
  }
}

REGISTER_OP(OperatorConf::kOutputConf, OutputOp);
REGISTER_OP_SAME_OUTPUT_BLOB_MEM_BLOCK_NUM(OperatorConf::kOutputConf, 1);
REGISTER_INTERFACE_OP(OperatorConf::kOutputConf);

}  // namespace oneflow
