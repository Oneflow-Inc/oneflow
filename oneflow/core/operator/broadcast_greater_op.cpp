#include "oneflow/core/operator/broadcast_greater_op.h"

namespace oneflow {

const PbMessage& BroadcastGreaterOp::GetCustomizedConf() const {
  return op_conf().broadcast_greater_conf();
}

REGISTER_OP(OperatorConf::kBroadcastGreaterConf, BroadcastGreaterOp);

}  // namespace oneflow
