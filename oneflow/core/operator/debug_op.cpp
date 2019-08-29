#include "oneflow/core/operator/debug_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void DebugOp::InitFromOpConf() {
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

Maybe<void> DebugOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                    const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("in");
  return Maybe<void>::Ok();
}

void DebugOp::GetSbpSignatures(
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  int64_t num_axes = LogicalBlobDesc4Ibn(input_bns().Get(0)).shape().NumAxes();
  SbpSignatureBuilder()
      .Split(input_bns(), 0)
      .Split(output_bns(), 0)
      .MakeSplitSignatureListBuilder(num_axes)
      .Build(sbp_sig_list);
}

REGISTER_CPU_OP(OperatorConf::kDebugConf, DebugOp);

}  // namespace oneflow
