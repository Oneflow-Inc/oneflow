#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  CHECK(op.op_conf().has_logistic_conf());
  if (DiffLbi4BnInOp("in") != nullptr) { *DiffLbi4BnInOp("in") = *DiffLbi4BnInOp("out"); }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kLogisticConf, &GenerateBackwardOpConf);

}  // namespace oneflow
