#ifndef ONEFLOW_CORE_OPERATOR_BROADCAST_ADD_OP_H_
#define ONEFLOW_CORE_OPERATOR_BROADCAST_ADD_OP_H_

#include "oneflow/core/operator/broadcast_binary_op.h"

namespace oneflow {

class BroadcastAddOp final : public BroadcastBinaryOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastAddOp);
  BroadcastAddOp() = default;
  ~BroadcastAddOp() = default;

  bool NeedInBlobWhenBackward() const override { return false; }
  bool NeedOutBlobWhenBackward() const override { return false; }

  const PbMessage& GetCustomizedConf() const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_BROADCAST_ADD_OP_H_
