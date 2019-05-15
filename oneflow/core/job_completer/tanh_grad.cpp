#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  CHECK(op.op_conf().has_tanh_conf());
  if (DiffLbi4BnInOp("in") != nullptr) {
    OperatorConf tanh_grad_op;
    tanh_grad_op.set_name(op.op_name() + "_grad");
    TanHGradOpConf* tanh_grad_op_conf = tanh_grad_op.mutable_tanh_grad_conf();
    tanh_grad_op_conf->set_y(GenLogicalBlobName(op.BnInOp2Lbi("out")));
    tanh_grad_op_conf->set_dy(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    tanh_grad_op_conf->set_dx("dx");
    op_confs->push_back(tanh_grad_op);
    DiffLbi4BnInOp("in")->set_op_name(tanh_grad_op.name());
    DiffLbi4BnInOp("in")->set_blob_name("dx");
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kTanhConf, &GenerateBackwardOpConf);

}  // namespace oneflow
