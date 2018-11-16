#include "oneflow/core/operator/broadcast_sub_op.h"

namespace oneflow {

const PbMessage& BroadcastSubOp::GetCustomizedConf() const {
  return op_conf().broadcast_sub_conf();
}

REGISTER_OP(OperatorConf::kBroadcastSubConf, BroadcastSubOp);

}  // namespace oneflow
