#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  CHECK(op.op_conf().has_gelu_conf());
  if (DiffLbi4BnInOp("in") != nullptr) {
    OperatorConf gelu_grad_op;
    gelu_grad_op.set_name(op.op_name() + "_grad");
    GeluGradOpConf* gelu_grad_op_conf = gelu_grad_op.mutable_gelu_grad_conf();
    gelu_grad_op_conf->set_x(GenLogicalBlobName(op.BnInOp2Lbi("in")));
    gelu_grad_op_conf->set_dy(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    gelu_grad_op_conf->set_dx("dx");
    op_confs->push_back(gelu_grad_op);
    DiffLbi4BnInOp("in")->set_op_name(gelu_grad_op.name());
    DiffLbi4BnInOp("in")->set_blob_name("dx");
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kGeluConf, &GenerateBackwardOpConf);

}  // namespace oneflow
