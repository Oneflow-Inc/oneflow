#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

void GenerateNvtxRangeBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  CHECK(op.op_conf().has_nvtx_range_push_conf() || op.op_conf().has_nvtx_range_pop_conf());
  if (DiffLbi4BnInOp("in") != nullptr) {
    DiffLbi4BnInOp("in")->set_op_name(DiffLbi4BnInOp("out")->op_name());
    DiffLbi4BnInOp("in")->set_blob_name(DiffLbi4BnInOp("out")->blob_name());
  }
}

REGISTER_OP_GRAD(OperatorConf::kNvtxRangePushConf, &GenerateNvtxRangeBackwardOpConf);
REGISTER_OP_GRAD(OperatorConf::kNvtxRangePopConf, &GenerateNvtxRangeBackwardOpConf);

}  // namespace

}  // namespace oneflow
