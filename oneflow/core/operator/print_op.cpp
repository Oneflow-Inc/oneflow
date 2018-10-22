#include "oneflow/core/operator/print_op.h"

namespace oneflow {

void PrintOp::InitFromOpConf() {
  CHECK(op_conf().has_print_conf());
  const PrintOpConf& conf = op_conf().print_conf();

  FOR_RANGE(int32_t, i, 0, conf.in_size()) { EnrollInputBn("in_" + std::to_string(i), false); }
}

const PbMessage& PrintOp::GetCustomizedConf() const { return op_conf().print_conf(); }

LogicalBlobId PrintOp::Lbi4InputBn(const std::string& input_bn) const {
  CHECK_STREQ(input_bn.substr(0, 3).c_str(), "in_");
  return GenLogicalBlobId(
      op_conf().print_conf().in(oneflow_cast<int32_t>(input_bn.substr(3))).lbn());
}

LogicalBlobId PrintOp::ibn2lbi(const std::string& input_bn) const { return Lbi4InputBn(input_bn); }

LogicalBlobId PrintOp::pibn2lbi(const std::string& input_bn) const {
  LogicalBlobId lbi = Lbi4InputBn(input_bn);
  lbi.set_is_pb_blob(true);
  return lbi;
}

REGISTER_OP(OperatorConf::kPrintConf, PrintOp);

}  // namespace oneflow
