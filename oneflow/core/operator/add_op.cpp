#include "oneflow/core/operator/add_op.h"

namespace oneflow {

void AddOp::VirtualInitFromOpConf() { CHECK(op_conf().has_add_conf()); }
const PbMessage& AddOp::GetCustomizedConf() const { return op_conf().add_conf(); }

bool AddOp::NeedOutWhenBackward() const {
  ActivationType activation = static_cast<ActivationType>(GetEnumFromCustomizedConf("activation"));
  if (activation != ActivationType::kNone) {
    return true;
  } else {
    return false;
  }
}

REGISTER_OP(OperatorConf::kAddConf, AddOp);

}  // namespace oneflow
