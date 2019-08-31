#include "oneflow/core/operator/sqrt_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void SqrtOp::InitFromOpConf() {
  CHECK(op_conf().has_sqrt_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out")->set_mutable_inplace_ibn("in");
}

const PbMessage& SqrtOp::GetCustomizedConf() const { return op_conf().sqrt_conf(); }

Maybe<void> SqrtOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                   const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("in");
  return Maybe<void>::Ok();
}

void SqrtOp::GetSbpSignatures(
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split(input_bns(), 0)
      .Split(output_bns(), 0)
      .MakeSplitSignatureListBuilder(LogicalBlobDesc4Ibn(output_bns().Get(0)).shape().NumAxes())
      .Build(sbp_sig_list);
}

REGISTER_OP(OperatorConf::kSqrtConf, SqrtOp);

}  // namespace oneflow
