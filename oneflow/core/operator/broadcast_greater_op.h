#ifndef ONEFLOW_CORE_OPERATOR_BROADCAST_GREATER_OP_H_
#define ONEFLOW_CORE_OPERATOR_BROADCAST_GREATER_OP_H_

#include "oneflow/core/operator/broadcast_binary_op.h"

namespace oneflow {

class BroadcastGreaterOp final : public BroadcastBinaryOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastGreaterOp);
  BroadcastGreaterOp() = default;
  ~BroadcastGreaterOp() = default;

  const PbMessage& GetCustomizedConf() const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_BROADCAST_GREATER_OP_H_
