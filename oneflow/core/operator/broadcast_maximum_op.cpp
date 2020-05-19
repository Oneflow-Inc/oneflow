#include "oneflow/core/operator/broadcast_binary_op.h"

namespace oneflow {

class BroadcastMaximumOp final : public BroadcastBinaryOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastMaximumOp);
  BroadcastMaximumOp() = default;
  ~BroadcastMaximumOp() override = default;

 private:
  const PbMessage& GetCustomizedConf() const override;
};

const PbMessage& BroadcastMaximumOp::GetCustomizedConf() const {
  return op_conf().broadcast_maximum_conf();
}

REGISTER_OP(OperatorConf::kBroadcastMaximumConf, BroadcastMaximumOp);

}  // namespace oneflow
