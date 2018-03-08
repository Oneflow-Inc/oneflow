#include "oneflow/core/operator/add_op.h"

namespace oneflow {

const PbMessage& AddOp::GetCustomizedConf() const {
  return op_conf().add_conf();
}

REGISTER_OP(OperatorConf::kAddConf, AddOp);

}  // namespace oneflow
