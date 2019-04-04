#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  CHECK(op.op_conf().has_add_conf());
  const AddOpConf& conf = op.op_conf().add_conf();
  FOR_RANGE(int32_t, i, 0, conf.in_size()) {
    const std::string& ibn = op.input_bns().Get(i);
    if (DiffLbi4BnInOp(ibn) != nullptr) { *DiffLbi4BnInOp(ibn) = *DiffLbi4BnInOp("out"); }
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kAddConf, &GenerateBackwardOpConf);

}  // namespace oneflow
