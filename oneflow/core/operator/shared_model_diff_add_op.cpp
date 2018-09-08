#include "oneflow/core/operator/shared_model_diff_add_op.h"

namespace oneflow {

void SharedModelDiffAddOp::VirtualInitFromOpConf() {
  CHECK(op_conf().has_shared_model_diff_add_conf());
}
const PbMessage& SharedModelDiffAddOp::GetCustomizedConf() const {
  return op_conf().shared_model_diff_add_conf();
}

REGISTER_OP(OperatorConf::kSharedModelDiffAddConf, SharedModelDiffAddOp);

}  // namespace oneflow
