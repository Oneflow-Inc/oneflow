#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  CHECK(op.op_conf().has_leaky_relu_conf());
  if (DiffLbi4BnInOp("in") != nullptr) {
    OperatorConf leaky_relu_grad_op;
    leaky_relu_grad_op.set_name(op.op_name() + "_grad");
    LeakyReluGradOpConf* leaky_relu_grad_op_conf = leaky_relu_grad_op.mutable_leaky_relu_grad_conf();
    leaky_relu_grad_op_conf->set_x(GenLogicalBlobName(op.BnInOp2Lbi("in")));
    leaky_relu_grad_op_conf->set_dy(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    leaky_relu_grad_op_conf->set_alpha(op.op_conf().leaky_relu_conf().alpha());
    leaky_relu_grad_op_conf->set_dx("dx");
    op_confs->push_back(leaky_relu_grad_op);
    DiffLbi4BnInOp("in")->set_op_name(leaky_relu_grad_op.name());
    DiffLbi4BnInOp("in")->set_blob_name(leaky_relu_grad_op_conf->dx());
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kLeakyReluConf, &GenerateBackwardOpConf);

}  // namespace oneflow
