#include "oneflow/core/operator/reshape_like_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void ReshapeLikeOp::InitFromOpConf() {
  CHECK(op_conf().has_reshape_like_conf());
  EnrollInputBn("x");
  EnrollOutputBn("y");
  EnrollInputBn("like", false)->set_use_header_only(true);
}

const PbMessage& ReshapeLikeOp::GetCustomizedConf() const { return op_conf().reshape_like_conf(); }

Maybe<void> ReshapeLikeOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  CHECK_EQ_OR_RETURN(GetBlobDesc4BnInOp("x")->shape().elem_cnt(),
                     GetBlobDesc4BnInOp("like")->shape().elem_cnt());
  GetBlobDesc4BnInOp("y")->CopyMetaFrom(*GetBlobDesc4BnInOp("like"));
  return Maybe<void>::Ok();
}

void ReshapeLikeOp::GetSbpSignatures(
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split(input_bns(), 0)
      .Split(output_bns(), 0)
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  SbpSignatureBuilder()
      .PartialSum(input_bns())
      .PartialSum(output_bns())
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
}

REGISTER_OP(OperatorConf::kReshapeLikeConf, ReshapeLikeOp);

}  // namespace oneflow
