#include "oneflow/core/operator/broadcast_not_equal_op.h"

namespace oneflow {

const PbMessage& BroadcastNotEqualOp::GetCustomizedConf() const {
  return op_conf().broadcast_not_equal_conf();
}

REGISTER_OP(OperatorConf::kBroadcastNotEqualConf, BroadcastNotEqualOp);

}  // namespace oneflow
