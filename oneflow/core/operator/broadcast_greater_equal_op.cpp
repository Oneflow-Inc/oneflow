#include "oneflow/core/operator/broadcast_greater_equal_op.h"

namespace oneflow {

const PbMessage& BroadcastGreaterEqualOp::GetCustomizedConf() const {
  return op_conf().broadcast_greater_equal_conf();
}

REGISTER_OP(OperatorConf::kBroadcastGreaterEqualConf, BroadcastGreaterEqualOp);

}  // namespace oneflow
