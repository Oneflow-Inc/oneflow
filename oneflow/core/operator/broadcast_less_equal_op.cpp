#include "oneflow/core/operator/broadcast_less_equal_op.h"

namespace oneflow {

const PbMessage& BroadcastLessEqualOp::GetCustomizedConf() const {
  return op_conf().broadcast_less_equal_conf();
}

REGISTER_OP(OperatorConf::kBroadcastLessEqualConf, BroadcastLessEqualOp);

}  // namespace oneflow
