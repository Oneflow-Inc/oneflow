#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  CHECK(op.op_conf().has_exp_conf());
  if (DiffLbi4BnInOp("in") != nullptr) {
    OperatorConf grad_op{};
    grad_op.set_name("System-AutoGrad-" + op.op_name());
    ExpGradOpConf* exp_grad_op_conf = grad_op.mutable_exp_grad_conf();
    exp_grad_op_conf->set_y(GenLogicalBlobName(op.BnInOp2Lbi("out")));
    exp_grad_op_conf->set_dy(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    exp_grad_op_conf->set_dx("dx");
    op_confs->push_back(grad_op);
    DiffLbi4BnInOp("in")->set_op_name(grad_op.name());
    DiffLbi4BnInOp("in")->set_blob_name(exp_grad_op_conf->dx());
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kExpConf, &GenerateBackwardOpConf);

}  // namespace oneflow
