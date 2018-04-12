#include "oneflow/core/operator/print_op.h"

namespace oneflow {

void PrintOp::InitFromOpConf() {
  CHECK(op_conf().has_print_conf());
  const PrintOpConf& conf = op_conf().print_conf();
  EnrollRepeatedInputBn("in", conf.in_size(), false);
}

const PbMessage& PrintOp::GetCustomizedConf() const {
  return op_conf().print_conf();
}

std::string PrintOp::ibn2lbn(const std::string& input_bn) const {
  CHECK(input_bn.substr(0, 3) == "in_");
  return op_conf()
      .print_conf()
      .in(oneflow_cast<int32_t>(input_bn.substr(3)))
      .lbn();
}

REGISTER_OP(OperatorConf::kPrintConf, PrintOp);

}  // namespace oneflow
