#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/operator/input_op.h"
#include "oneflow/core/operator/interface_op_util.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void InputOp::InitFromOpConf() {
  CHECK(op_conf().has_input_conf());
  if (op_conf().input_conf().has_tick()) { EnrollInputBn("tick", false); }
  EnrollOutputBn("out", false);
}

const PbMessage& InputOp::GetCustomizedConf() const { return op_conf().input_conf(); }

Maybe<void> InputOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                    const ParallelContext* parallel_ctx,
                                    const SbpSignature* sbp_signature) const {
  return InterfaceOpUtil::InferOutBlobDesc(op_conf().input_conf().blob_conf(),
                                           GetBlobDesc4BnInOp("out"), parallel_ctx);
}

Maybe<void> InputOp::InferBatchAxis(
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  *BatchAxis4BnInOp("out") = op_conf().input_conf().blob_conf().batch_axis();
  return Maybe<void>::Ok();
}

Maybe<void> InputOp::InferSbpSignature(
    SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
    const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
    std::function<Maybe<const SbpInferHint*>(const std::string&)> SbpInferHint4Ibn,
    const ParallelDesc& parallel_desc) const {
  SbpSignatureList sbp_sig_list;
  JUST(GetSbpSignatures(&sbp_sig_list));
  *sbp_signature = sbp_sig_list.sbp_signature(0);
  return Maybe<void>::Ok();
}

Maybe<void> InputOp::GetSbpSignatures(SbpSignatureList* sbp_sig_list) const {
  InterfaceOpUtil::GetInputLikeOpSbpSignature(op_conf().input_conf().blob_conf(), input_bns(),
                                              output_bns(),
                                              sbp_sig_list->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kInputConf, InputOp);
REGISTER_OP_SAME_OUTPUT_BLOB_REGST_NUM(OperatorConf::kInputConf, 1);
REGISTER_INTERFACE_OP(OperatorConf::kInputConf);

}  // namespace oneflow
