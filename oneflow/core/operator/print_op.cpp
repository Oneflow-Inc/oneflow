#include "oneflow/core/operator/print_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void PrintOp::InitFromOpConf() {
  CHECK(op_conf().has_print_conf());
  const PrintOpConf& conf = op_conf().print_conf();
  EnrollRepeatedInputBn("in", conf.in_size(), false);
}

const PbMessage& PrintOp::GetCustomizedConf() const { return op_conf().print_conf(); }

LogicalBlobId PrintOp::ibn2lbi(const std::string& input_bn) const {
  CHECK_STREQ(input_bn.substr(0, 3).c_str(), "in_");
  return GenLogicalBlobId(
      op_conf().print_conf().in(oneflow_cast<int32_t>(input_bn.substr(3))).lbn());
}

Maybe<void> PrintOp::GetSbpSignatures(SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split(input_bns(), 0)
      .Split(output_bns(), 0)
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

REGISTER_CPU_OP(OperatorConf::kPrintConf, PrintOp);

}  // namespace oneflow
