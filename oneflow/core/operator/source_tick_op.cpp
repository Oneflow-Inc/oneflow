#include "oneflow/core/operator/source_tick_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void SourceTickOp::InitFromOpConf() {
  CHECK(op_conf().has_source_tick_conf());
  CHECK(op_conf().ctrl_in_op_name().empty());
  EnrollOutputBn("out", false);
}

LogicalNode* SourceTickOp::NewProperLogicalNode() const { return new SourceTickLogicalNode(); }

const PbMessage& SourceTickOp::GetCustomizedConf() const { return op_conf().source_tick_conf(); }

Maybe<void> SourceTickOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  CHECK_EQ_OR_RETURN(parallel_ctx->parallel_num(), 1);
  GetBlobDesc4BnInOp("out")->mut_shape() = Shape({1});
  return Maybe<void>::Ok();
}

Maybe<void> SourceTickOp::InferBatchAxis(
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  BatchAxis4BnInOp("out")->clear_value();
  return Maybe<void>::Ok();
}

Maybe<void> SourceTickOp::GetSbpSignatures(SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder().Split(output_bns(), 0).Build(sbp_sig_list->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

REGISTER_CPU_OP(OperatorConf::kSourceTickConf, SourceTickOp);
REGISTER_TICK_TOCK_OP(OperatorConf::kSourceTickConf);

}  // namespace oneflow
