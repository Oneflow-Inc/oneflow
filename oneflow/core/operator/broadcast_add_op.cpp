#include "oneflow/core/operator/broadcast_add_op.h"

namespace oneflow {

const PbMessage& BroadcastAddOp::GetCustomizedConf() const {
  return op_conf().broadcast_add_conf();
}

REGISTER_OP(OperatorConf::kBroadcastAddConf, BroadcastAddOp);

}  // namespace oneflow
