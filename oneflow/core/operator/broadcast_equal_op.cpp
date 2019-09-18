#include "oneflow/core/operator/broadcast_equal_op.h"

namespace oneflow {

const PbMessage& BroadcastEqualOp::GetCustomizedConf() const {
  return op_conf().broadcast_equal_conf();
}

REGISTER_OP(OperatorConf::kBroadcastEqualConf, BroadcastEqualOp);

}  // namespace oneflow
