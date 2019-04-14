#include "oneflow/core/operator/broadcast_like_op.h"

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

REGISTER_OP(OperatorConf::kBroadcastLikeConf, BroadcastLikeOp);

}  // namespace oneflow
