#include "oneflow/core/operator/print_op.h"

namespace oneflow {

void PrintOp::InitFromOpConf() {
  CHECK(op_conf().has_print_conf());
  EnrollRepeatedInputBn("lbn", false);
}

const PbMessage& PrintOp::GetCustomizedConf() const {
  return op_conf().print_conf();
}

REGISTER_OP(OperatorConf::kPrintConf, PrintOp);

}  // namespace oneflow
