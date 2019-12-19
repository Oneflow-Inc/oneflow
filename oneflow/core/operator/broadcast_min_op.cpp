#include "oneflow/core/operator/broadcast_binary_op.h"

namespace oneflow {

class BroadcastMinOp final : public BroadcastBinaryOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastMinOp);
  BroadcastMinOp() = default;
  ~BroadcastMinOp() = default;

  const PbMessage& GetCustomizedConf() const override { return op_conf().broadcast_min_conf(); }
};

REGISTER_OP(OperatorConf::kBroadcastMinConf, BroadcastMinOp);

}  // namespace oneflow
