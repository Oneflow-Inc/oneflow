#include "oneflow/core/operator/broadcast_like_op.h"

namespace oneflow {

void BroadcastLikeOp::InitFromOpConf() {
  EnrollInputBn("a");
  EnrollInputBn("b")->set_use_header_only(true);
  EnrollOutputBn("out");
}

const PbMessage& BroadcastLikeOp::GetCustomizedConf() const {
  return op_conf().broadcast_like_conf();
}

REGISTER_OP(OperatorConf::kBroadcastLikeConf, BroadcastLikeOp);

}  // namespace oneflow
