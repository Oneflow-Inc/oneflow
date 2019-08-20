#ifndef ONEFLOW_CORE_OPERATOR_BROADCAST_SUB_OP_H_
#define ONEFLOW_CORE_OPERATOR_BROADCAST_SUB_OP_H_

#include "oneflow/core/operator/broadcast_binary_op.h"

namespace oneflow {

class BroadcastSubOp final : public BroadcastBinaryOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastSubOp);
  BroadcastSubOp() = default;
  ~BroadcastSubOp() = default;

  const PbMessage& GetCustomizedConf() const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_BROADCAST_SUB_OP_H_
