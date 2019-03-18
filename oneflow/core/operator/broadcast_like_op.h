#ifndef ONEFLOW_CORE_OPERATOR_BROADCAST_LIKE_OP_H_
#define ONEFLOW_CORE_OPERATOR_BROADCAST_LIKE_OP_H_

#include "oneflow/core/operator/broadcast_binary_op.h"

namespace oneflow {

class BroadcastLikeOp final : public BroadcastBinaryOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastLikeOp);
  BroadcastLikeOp() = default;
  ~BroadcastLikeOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_BROADCAST_LIKE_OP_H_
