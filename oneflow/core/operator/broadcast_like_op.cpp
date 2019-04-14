#include "oneflow/core/operator/broadcast_like_op.h"
#include "oneflow/core/operator/reduce_sbp_signature_rule.h"

namespace oneflow {

void BroadcastLikeOp::InitFromOpConf() {
  EnrollInputBn("x");
  EnrollInputBn("like", false)->set_use_header_only(true);
  EnrollOutputBn("y");
}

const PbMessage& BroadcastLikeOp::GetCustomizedConf() const {
  return op_conf().broadcast_like_conf();
}

void BroadcastLikeOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  GetBlobDesc4BnInOp("y")->CopyMetaFrom(*GetBlobDesc4BnInOp("like"));
}

void BroadcastLikeOp::GetSbpSignatureRules(
    std::vector<std::unique_ptr<const SbpSignatureRule>>* rules) const {
  const auto& reduced_axes = op_conf().broadcast_like_conf().reduced_axis();
  GetReduceGradSbpSignatureRules(this, "like", {reduced_axes.begin(), reduced_axes.end()}, rules);
}

REGISTER_OP(OperatorConf::kBroadcastLikeConf, BroadcastLikeOp);

}  // namespace oneflow
