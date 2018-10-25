#include "oneflow/core/operator/log_counter_op.h"

namespace oneflow {

void LogCounterOp::InitFromOpConf() {
  CHECK(op_conf().has_log_counter_conf());
  EnrollInputBn("in", false);
}

const PbMessage& LogCounterOp::GetCustomizedConf() const { return op_conf().log_counter_conf(); }

REGISTER_OP(OperatorConf::kLogCounterConf, LogCounterOp);

}  // namespace oneflow
