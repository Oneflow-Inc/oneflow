#include "oneflow/core/operator/broadcast_binary_op.h"

namespace oneflow {

class BroadcastEqualOp final : public BroadcastBinaryOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastEqualOp);
  BroadcastEqualOp() = default;
  ~BroadcastEqualOp() = default;

  const PbMessage& GetCustomizedConf() const override { return op_conf().broadcast_equal_conf(); }
};

REGISTER_OP(OperatorConf::kBroadcastEqualConf, BroadcastEqualOp);

}  // namespace oneflow
