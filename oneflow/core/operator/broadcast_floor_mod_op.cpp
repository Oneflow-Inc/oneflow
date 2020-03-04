#include "oneflow/core/operator/broadcast_binary_op.h"

namespace oneflow {

class BroadcastFloorModOp final : public BroadcastBinaryOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastFloorModOp);
  BroadcastFloorModOp() = default;
  ~BroadcastFloorModOp() override = default;

 private:
  const PbMessage& GetCustomizedConf() const override;
};

const PbMessage& BroadcastFloorModOp::GetCustomizedConf() const {
  return op_conf().broadcast_floor_mod_conf();
}

REGISTER_OP(OperatorConf::kBroadcastFloorModConf, BroadcastFloorModOp);

}  // namespace oneflow
