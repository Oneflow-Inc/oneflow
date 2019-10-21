#include "oneflow/core/operator/dropout_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void DropoutOp::InitFromOpConf() {
  CHECK_GT(op_conf().dropout_conf().scale(), 1);
  EnrollInputBn("in");
  EnrollInputBn("mask", false);
  EnrollOutputBn("out");
}

const PbMessage& DropoutOp::GetCustomizedConf() const { return op_conf().dropout_conf(); }

Maybe<void> DropoutOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("in");
  CHECK_OR_RETURN(GetBlobDesc4BnInOp("mask")->shape() == GetBlobDesc4BnInOp("in")->shape());
  return Maybe<void>::Ok();
}

Maybe<void> DropoutOp::InferBatchAxis(
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  for (const auto& obn : output_bns()) { *BatchAxis4BnInOp(obn) = *BatchAxis4BnInOp("in"); }
  return Maybe<void>::Ok();
}

Maybe<void> DropoutOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split(input_bns(), 0)
      .Split(output_bns(), 0)
      .MakeSplitSignatureListBuilder(JUST(LogicalBlobDesc4Ibn("in"))->shape().NumAxes())
      .Build(sbp_sig_list);
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kDropoutConf, DropoutOp);

}  // namespace oneflow
