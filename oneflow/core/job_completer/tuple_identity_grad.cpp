#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  FOR_RANGE(int32_t, i, 0, op.input_bns().size()) {
    LogicalBlobId* in_diff_lbi = DiffLbi4BnInOp(op.input_bns().Get(i));
    if (in_diff_lbi != nullptr) { *in_diff_lbi = *DiffLbi4BnInOp(op.output_bns().Get(i)); }
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kTupleIdentityConf, &GenerateBackwardOpConf);

}  // namespace oneflow
