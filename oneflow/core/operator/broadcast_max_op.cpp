#include "oneflow/core/operator/broadcast_binary_op.h"

namespace oneflow {

class BroadcastMaxOp final : public BroadcastBinaryOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastMaxOp);
  BroadcastMaxOp() = default;
  ~BroadcastMaxOp() = default;

  const PbMessage& GetCustomizedConf() const override { return op_conf().broadcast_max_conf(); }
};

REGISTER_OP(OperatorConf::kBroadcastMaxConf, BroadcastMaxOp);

}  // namespace oneflow
