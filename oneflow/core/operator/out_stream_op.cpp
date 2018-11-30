#include "oneflow/core/operator/out_stream_op.h"

namespace oneflow {

void OutStreamOp::InitFromOpConf() {
  CHECK(op_conf().has_out_stream_conf());
  const OutStreamOpConf& conf = op_conf().out_stream_conf();
  EnrollRepeatedInputBn("in", conf.in_size(), false);
}

const PbMessage& OutStreamOp::GetCustomizedConf() const { return op_conf().out_stream_conf(); }

LogicalBlobId OutStreamOp::ibn2lbi(const std::string& input_bn) const {
  CHECK_STREQ(input_bn.substr(0, 3).c_str(), "in_");
  return GenLogicalBlobId(
      op_conf().out_stream_conf().in(oneflow_cast<int32_t>(input_bn.substr(3))).lbn());
}

REGISTER_CPU_OP(OperatorConf::kOutStreamConf, OutStreamOp);

}  // namespace oneflow
