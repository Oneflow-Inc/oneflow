#ifndef ONEFLOW_CORE_OPERATOR_BROADCAST_GREATER_EQUAL_OP_H_
#define ONEFLOW_CORE_OPERATOR_BROADCAST_GREATER_EQUAL_OP_H_

#include "oneflow/core/operator/broadcast_binary_op.h"

namespace oneflow {

class BroadcastGreaterEqualOp final : public BroadcastBinaryOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastGreaterEqualOp);
  BroadcastGreaterEqualOp() = default;
  ~BroadcastGreaterEqualOp() = default;

  const PbMessage& GetCustomizedConf() const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_BROADCAST_GREATER_EQUAL_OP_H_
