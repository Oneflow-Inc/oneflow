#include "oneflow/core/operator/broadcast_div_op.h"

namespace oneflow {

const PbMessage& BroadcastDivOp::GetCustomizedConf() const {
  return op_conf().broadcast_div_conf();
}

REGISTER_OP(OperatorConf::kBroadcastDivConf, BroadcastDivOp);

}  // namespace oneflow
