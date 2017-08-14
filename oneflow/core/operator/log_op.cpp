#include "oneflow/core/operator/log_op.h"

namespace oneflow {

void LogOp::InitFromOpConf(const OperatorConf& op_conf) {
  CHECK(op_conf.has_log_conf());
  mut_op_conf() = op_conf;
  int32_t i = 0;
  for (const std::string& lbn : op_conf.log_conf().lbn()) {
    std::string ibn = "in_" + std::to_string(i++);
    CHECK(ibn2lbn_.emplace(ibn, lbn).second);
    EnrollInputBn(ibn, false);
  }
}

const PbMessage& LogOp::GetSpecialConf() const { return op_conf().log_conf(); }

REGISTER_OP(OperatorConf::kLogConf, LogOp);

}  // namespace oneflow
