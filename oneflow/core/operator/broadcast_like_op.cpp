#include "oneflow/core/operator/broadcast_like_op.h"
#include "oneflow/core/operator/reduce_sbp_util.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void BroadcastLikeOp::InitFromOpConf() {
  EnrollInputBn("x");
  EnrollInputBn("like", false)->set_use_header_only(true);
  EnrollOutputBn("y");
}

const PbMessage& BroadcastLikeOp::GetCustomizedConf() const {
  return op_conf().broadcast_like_conf();
}

Maybe<void> BroadcastLikeOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  GetBlobDesc4BnInOp("y")->CopyMetaFrom(*GetBlobDesc4BnInOp("like"));
  return Maybe<void>::Ok();
}

Maybe<void> BroadcastLikeOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  int32_t num_axes = JUST(LogicalBlobDesc4Ibn("like"))->shape().NumAxes();
  auto IsReducedAxis = ReduceSbpUtil::MakePredicatorIsReducedAxis(
      op_conf().broadcast_like_conf().reduced_axis(), num_axes);
  FOR_RANGE(int64_t, i, 0, num_axes) {
    if (IsReducedAxis(i)) {
      SbpSignatureBuilder()
          .Broadcast("x")
          .Split("like", i)
          .Split(output_bns(), i)
          .Build(sbp_sig_list->mutable_sbp_signature()->Add());
    } else {
      SbpSignatureBuilder()
          .Split(input_bns(), i)
          .Split(output_bns(), i)
          .Build(sbp_sig_list->mutable_sbp_signature()->Add());
    }
  }
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kBroadcastLikeConf, BroadcastLikeOp);

}  // namespace oneflow
