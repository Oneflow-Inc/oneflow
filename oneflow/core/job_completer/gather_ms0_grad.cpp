#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

Maybe<void> GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp,
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4BnInOp) {
  CHECK(op.op_conf().has_gather_ms0_conf());
  if (DiffLbi4BnInOp("in") != nullptr) {
    const BlobDesc& in_logical_blob_desc = LogicalBlobDesc4BnInOp("in");
    const int64_t gather_dim_size = in_logical_blob_desc.shape().At(0);
    OperatorConf gather_ms0_grad_op;
    gather_ms0_grad_op.set_name("System-AutoGrad-" + op.op_name());
    GatherMs0GradOpConf* conf = gather_ms0_grad_op.mutable_gather_ms0_grad_conf();
    conf->set_gather_dim_size(gather_dim_size);
    conf->set_indices(GenLogicalBlobName(op.BnInOp2Lbi("indices")));
    conf->set_out_diff(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    conf->set_in_diff("in_diff");
    op_confs->push_back(gather_ms0_grad_op);
    DiffLbi4BnInOp("in")->set_op_name(gather_ms0_grad_op.name());
    DiffLbi4BnInOp("in")->set_blob_name(conf->in_diff());
  }
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kGatherMs0Conf, &GenerateBackwardOpConf);

}  // namespace oneflow
