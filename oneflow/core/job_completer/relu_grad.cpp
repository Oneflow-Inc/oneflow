#include "oneflow/core/autograd/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  CHECK(op.op_conf().has_relu_conf());
  if (DiffLbi4BnInOp("in") != nullptr) {
    OperatorConf relu_grad_op;
    relu_grad_op.set_name(op.op_name() + "_grad");
    ReluGradOpConf* relu_grad_op_conf = relu_grad_op.mutable_relu_grad_conf();
    relu_grad_op_conf->set_y(GenLogicalBlobName(op.BnInOp2Lbi("out")));
    relu_grad_op_conf->set_dy(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    relu_grad_op_conf->set_dx("dx");
    op_confs->push_back(relu_grad_op);
    DiffLbi4BnInOp("in")->set_op_name(relu_grad_op.name());
    DiffLbi4BnInOp("in")->set_blob_name("dx");
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kReluConf, &GenerateBackwardOpConf);

}  // namespace oneflow
