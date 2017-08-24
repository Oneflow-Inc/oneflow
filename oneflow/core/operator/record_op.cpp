#include "oneflow/core/operator/record_op.h"

namespace oneflow {

void RecordOp::InitFromOpConf(const OperatorConf& op_conf) {
  CHECK(op_conf.has_record_conf());
  mut_op_conf() = op_conf;
  int32_t i = 0;
  for (const std::string& lbn : op_conf.record_conf().lbn()) {
    std::string ibn = "in_" + std::to_string(i++);
    CHECK(ibn2lbn_.emplace(ibn, lbn).second);
    EnrollInputBn(ibn, false);
  }
}

const PbMessage& RecordOp::GetSpecialConf() const {
  return op_conf().record_conf();
}

REGISTER_OP(OperatorConf::kRecordConf, RecordOp);

}  // namespace oneflow
