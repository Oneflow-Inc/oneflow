#include "oneflow/core/operator/reshape_like_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"
#include "oneflow/core/operator/reshape_op_util.h"

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

Maybe<void> ReshapeLikeOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    const ParallelDesc& parallel_desc, SbpSignatureList* sbp_sig_list) const {
  const auto& x_shape = JUST(LogicalBlobDesc4Ibn("x"))->shape();
  const auto& like_shape = JUST(LogicalBlobDesc4Ibn("like"))->shape();
  return ReshapeOpUtil::GetReshapeSbpSignatures(
      x_shape, like_shape, StdVec2PbRpf<std::string>({"x"}),
      StdVec2PbRpf<std::string>({"like", "y"}), parallel_desc.parallel_num(), sbp_sig_list);
}

REGISTER_OP(OperatorConf::kReshapeLikeConf, ReshapeLikeOp);

}  // namespace oneflow
