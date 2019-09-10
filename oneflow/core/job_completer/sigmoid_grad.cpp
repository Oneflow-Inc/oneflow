#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  CHECK(op.op_conf().has_sigmoid_conf());
  if (DiffLbi4BnInOp("in") != nullptr) {
    OperatorConf sigmoid_grad_op;
    sigmoid_grad_op.set_name(op.op_name() + "_grad");
    SigmoidGradOpConf* sigmoid_grad_op_conf = sigmoid_grad_op.mutable_sigmoid_grad_conf();
    sigmoid_grad_op_conf->set_y(GenLogicalBlobName(op.BnInOp2Lbi("out")));
    sigmoid_grad_op_conf->set_dy(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    sigmoid_grad_op_conf->set_dx("dx");
    op_confs->push_back(sigmoid_grad_op);
    DiffLbi4BnInOp("in")->set_op_name(sigmoid_grad_op.name());
    DiffLbi4BnInOp("in")->set_blob_name("dx");
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kSigmoidConf, &GenerateBackwardOpConf);

}  // namespace oneflow
