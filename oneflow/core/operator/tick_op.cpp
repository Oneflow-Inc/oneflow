#include "oneflow/core/operator/tick_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void TickOp::InitFromOpConf() {
  EnrollRepeatedInputBn("tick", false);
  EnrollOutputBn("out", false);
}

void TickOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx) const {
  GetBlobDesc4BnInOp("out")->mut_shape() = Shape({1});
}

void TickOp::InferHasBatchDim(std::function<bool*(const std::string&)> HasBatchDim4BnInOp) const {
  *HasBatchDim4BnInOp("out") = false;
}

void TickOp::GetSbpSignatures(SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split(input_bns(), 0)
      .Split(output_bns(), 0)
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
}

REGISTER_OP_SAME_OUTPUT_BLOB_MEM_BLOCK_NUM(OperatorConf::kTickConf, 2);
REGISTER_OP(OperatorConf::kTickConf, TickOp);

}  // namespace oneflow
