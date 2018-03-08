#include "oneflow/core/operator/add_op.h"

namespace oneflow {
void AddOp::InitFromOpConf() {
  CHECK(op_conf().has_add_conf());
  ElementwiseOp::InitFromOpConf();
}
const PbMessage& AddOp::GetCustomizedConf() const {
  return op_conf().add_conf();
}

REGISTER_OP(OperatorConf::kAddConf, AddOp);

}  // namespace oneflow
