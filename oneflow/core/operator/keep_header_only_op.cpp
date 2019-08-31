#include "oneflow/core/operator/keep_header_only_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void KeepHeaderOnlyOp::InitFromOpConf() {
  CHECK_EQ(GetPbRpfFromCustomizedConf<std::string>("in").size(),
           GetPbRpfFromCustomizedConf<std::string>("out").size());
  EnrollRepeatedInputBn("in", false);
  EnrollRepeatedOutputBn("out", false);
}

Maybe<void> KeepHeaderOnlyOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  size_t in_num = GetPbRpfFromCustomizedConf<std::string>("in").size();
  for (size_t i = 0; i < in_num; ++i) {
    BlobDesc* out = GetBlobDesc4BnInOp(GenRepeatedBn("out", i));
    *out = *GetBlobDesc4BnInOp(GenRepeatedBn("in", i));
    out->set_is_body_disabled(true);
  }
  return Maybe<void>::Ok();
}

void KeepHeaderOnlyOp::GetSbpSignatures(
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  int64_t num_axes = LogicalBlobDesc4Ibn(SoleIbn()).shape().NumAxes();
  SbpSignatureBuilder()
      .Split(input_bns(), 0)
      .Split(output_bns(), 0)
      .MakeSplitSignatureListBuilder(num_axes)
      .Build(sbp_sig_list);
  SbpSignatureBuilder()
      .PartialSum(input_bns())
      .PartialSum(output_bns())
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
}

REGISTER_OP(OperatorConf::kKeepHeaderOnlyConf, KeepHeaderOnlyOp);
REGISTER_OP_SAME_OUTPUT_BLOB_MEM_BLOCK_NUM(OperatorConf::kKeepHeaderOnlyConf, 100);
}  // namespace oneflow
