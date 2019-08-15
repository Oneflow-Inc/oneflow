#include "oneflow/core/operator/broadcast_less_op.h"

namespace oneflow {

const PbMessage& BroadcastLessOp::GetCustomizedConf() const {
  return op_conf().broadcast_less_conf();
}

REGISTER_OP(OperatorConf::kBroadcastLessConf, BroadcastLessOp);

}  // namespace oneflow
