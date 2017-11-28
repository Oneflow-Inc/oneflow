#include "oneflow/core/operator/print_op.h"

namespace oneflow {

void PrintOp::InitFromOpConf() {
  CHECK(op_conf().has_print_conf());
  int32_t i = 0;
  for (const std::string& lbn : op_conf().print_conf().lbn()) {
    std::string ibn = "in_" + std::to_string(i++);
    CHECK(ibn2lbn_.emplace(ibn, lbn).second);
    EnrollInputBn(ibn, false);
  }
}

const PbMessage& PrintOp::GetSpecialConf() const {
  return op_conf().print_conf();
}

REGISTER_OP(OperatorConf::kPrintConf, PrintOp);

}  // namespace oneflow
