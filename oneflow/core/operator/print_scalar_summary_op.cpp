#include "oneflow/core/operator/print_scalar_summary_op.h"

namespace oneflow {

void PrintScalarSummaryOp::InitFromOpConf() { EnrollInputBn("x"); }

void PrintScalarSummaryOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  CHECK_EQ(GetBlobDesc4BnInOp("x")->shape().NumAxes(), 1);
}

const PbMessage& PrintScalarSummaryOp::GetCustomizedConf() const {
  return op_conf().print_scalar_summary_conf();
}

void PrintScalarSummaryOp::GetSbpSignatures(
    std::vector<std::unique_ptr<const SbpSignature>>*) const {
  // do nothing. only unparalleled sbp signature supported
}

REGISTER_CPU_OP(OperatorConf::kPrintScalarSummaryConf, PrintScalarSummaryOp);

}  // namespace oneflow
