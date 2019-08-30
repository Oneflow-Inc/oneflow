#include "oneflow/core/operator/scalar_mul_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void ScalarMulOp::InitFromOpConf() {
  EnrollInputBn("in");
  EnrollOutputBn("out")->set_mutable_inplace_ibn("in");
}

Maybe<void> ScalarMulOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("in");
  return Maybe<void>::Ok();
}

void ScalarMulOp::GetSbpSignatures(
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split(input_bns(), 0)
      .Split(output_bns(), 0)
      .MakeSplitSignatureListBuilder(LogicalBlobDesc4Ibn("in").shape().NumAxes())
      .Build(sbp_sig_list);
  SbpSignatureBuilder()
      .PartialSum(input_bns())
      .PartialSum(output_bns())
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
}

REGISTER_OP(OperatorConf::kScalarMulConf, ScalarMulOp);

}  // namespace oneflow
