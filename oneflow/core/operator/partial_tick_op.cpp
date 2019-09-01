#include "oneflow/core/operator/partial_tick_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void PartialTickOp::InitFromOpConf() {
  CHECK(op_conf().has_partial_tick_conf());
  EnrollInputBn("tick", false);
  EnrollOutputBn("out", false);
}

Maybe<void> PartialTickOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  GetBlobDesc4BnInOp("out")->mut_shape() = Shape({1});
  return Maybe<void>::Ok();
}

Maybe<void> PartialTickOp::InferHasBatchDim(
    std::function<bool*(const std::string&)> HasBatchDim4BnInOp) const {
  *HasBatchDim4BnInOp("out") = false;
  return Maybe<void>::Ok();
}

void PartialTickOp::GetSbpSignatures(
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Broadcast(input_bns())
      .PartialSum(output_bns())
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
}

REGISTER_OP_SAME_OUTPUT_BLOB_MEM_BLOCK_NUM(OperatorConf::kPartialTickConf, 2);
REGISTER_OP(OperatorConf::kPartialTickConf, PartialTickOp);
REGISTER_TICK_TOCK_OP(OperatorConf::kPartialTickConf);

}  // namespace oneflow
