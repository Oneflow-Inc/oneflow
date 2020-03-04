#include "oneflow/core/operator/broadcast_binary_op.h"

namespace oneflow {

class BroadcastMinimumOp final : public BroadcastBinaryOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastMinimumOp);
  BroadcastMinimumOp() = default;
  ~BroadcastMinimumOp() override = default;

 private:
  const PbMessage& GetCustomizedConf() const override;
};

const PbMessage& BroadcastMinimumOp::GetCustomizedConf() const {
  return op_conf().broadcast_minimum_conf();
}

REGISTER_OP(OperatorConf::kBroadcastMinimumConf, BroadcastMinimumOp);

}  // namespace oneflow
