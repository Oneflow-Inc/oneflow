#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp,
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4BnInOp) {
  CHECK(op.op_conf().has_local_scatter_nd_update_conf());
  if (DiffLbi4BnInOp("in") != nullptr) {
    OperatorConf grad_op_conf;
    grad_op_conf.set_name("System-AutoGrad-" + op.op_name());
    auto* conf = grad_op_conf.mutable_local_gather_nd_conf();
    conf->set_in(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    conf->set_indices(GenLogicalBlobName(op.BnInOp2Lbi("indices")));
    conf->set_out("out");
    op_confs->push_back(grad_op_conf);
    DiffLbi4BnInOp("in")->set_op_name(grad_op_conf.name());
    DiffLbi4BnInOp("in")->set_blob_name(conf->out());
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kBatchGatherConf, &GenerateBackwardOpConf);

}  // namespace oneflow
