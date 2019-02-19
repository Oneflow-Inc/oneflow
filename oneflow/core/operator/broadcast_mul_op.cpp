#include "oneflow/core/operator/broadcast_mul_op.h"

namespace oneflow {

const PbMessage& BroadcastMulOp::GetCustomizedConf() const {
  return op_conf().broadcast_mul_conf();
}

REGISTER_OP(OperatorConf::kBroadcastMulConf, BroadcastMulOp);

}  // namespace oneflow
