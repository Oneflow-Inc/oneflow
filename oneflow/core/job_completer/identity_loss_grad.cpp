#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  auto* diff_lbi = DiffLbi4BnInOp("prediction");
  if (diff_lbi != nullptr) { *diff_lbi = *DiffLbi4BnInOp("loss"); }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kIdentityLossConf, &GenerateBackwardOpConf);

}  // namespace oneflow
