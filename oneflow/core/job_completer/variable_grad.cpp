#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  // do nothing
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kVariableConf, &GenerateBackwardOpConf);

}  // namespace oneflow
