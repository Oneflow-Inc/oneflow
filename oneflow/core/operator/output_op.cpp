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
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  if (in_blob_desc->is_dynamic()) {
    *out_blob_desc = *in_blob_desc;
  } else {
    InterfaceOpUtil::InferOutBlobDesc(op_conf().output_conf().blob_conf(), out_blob_desc,
                                      parallel_ctx);
    CHECK_OR_RETURN(*out_blob_desc == *in_blob_desc);
  }
  return Maybe<void>::Ok();
}

const PbMessage& OutputOp::GetCustomizedConf() const { return op_conf().output_conf(); }

Maybe<void> OutputOp::InferBatchAxis(
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  OptInt64* out_batch_axis = BatchAxis4BnInOp("out");
  InterfaceOpUtil::InferBatchAxis(op_conf().output_conf().blob_conf(), out_batch_axis);
  CHECK_OR_RETURN(*out_batch_axis == *BatchAxis4BnInOp("in"));
  return Maybe<void>::Ok();
}

Maybe<void> OutputOp::InferSbpSignature(
    SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
    const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
    std::function<Maybe<const SbpInferHint*>(const std::string&)> SbpInferHint4Ibn,
    const ParallelDesc& parallel_desc) const {
  InterfaceOpUtil::GetOutputLikeOpSbpSignature(op_conf().output_conf().blob_conf(), input_bns(),
                                               output_bns(), sbp_signature);
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kOutputConf, OutputOp);
REGISTER_OP_SAME_OUTPUT_BLOB_REGST_NUM(OperatorConf::kOutputConf, 1);
REGISTER_INTERFACE_OP(OperatorConf::kOutputConf);

}  // namespace oneflow
