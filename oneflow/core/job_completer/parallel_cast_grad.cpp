#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  LogicalBlobId* in_diff_lbi = DiffLbi4BnInOp("in");
  if (in_diff_lbi != nullptr) {
    const ParallelCastOpConf& conf = op.op_conf().parallel_cast_conf();
    const LogicalBlobId* out_diff_lbi = DiffLbi4BnInOp("out");
    if (conf.has_gradient_split_axis()) {
      OperatorConf grad_op{};
      grad_op.set_name("System-AutoGrad-" + op.op_name());
      ParallelCastOpConf* parallel_cast_conf = grad_op.mutable_parallel_cast_conf();
      *parallel_cast_conf->mutable_split_axis() = conf.gradient_split_axis();
      parallel_cast_conf->set_in(GenLogicalBlobName(*out_diff_lbi));
      parallel_cast_conf->set_out("out");
      in_diff_lbi->set_op_name(grad_op.name());
      in_diff_lbi->set_blob_name(parallel_cast_conf->out());
      op_confs->push_back(grad_op);
    } else {
      *in_diff_lbi = *out_diff_lbi;
    }
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kParallelCastConf, &GenerateBackwardOpConf);

}  // namespace oneflow
