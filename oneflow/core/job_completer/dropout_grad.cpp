#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  CHECK(op.op_conf().has_dropout_conf());
  if (DiffLbi4BnInOp("in") != nullptr) {
    OperatorConf dropout_grad_op;
    dropout_grad_op.set_name(op.op_name() + "_grad");
    DropoutGradOpConf* dropout_grad_op_conf = dropout_grad_op.mutable_dropout_grad_conf();
    dropout_grad_op_conf->set_mask(GenLogicalBlobName(op.BnInOp2Lbi("mask")));
    dropout_grad_op_conf->set_dy(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    dropout_grad_op_conf->set_dx("dx");
    dropout_grad_op_conf->set_scale(op.op_conf().dropout_conf().scale());
    op_confs->push_back(dropout_grad_op);
    DiffLbi4BnInOp("in")->set_op_name(dropout_grad_op.name());
    DiffLbi4BnInOp("in")->set_blob_name("dx");
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kDropoutConf, &GenerateBackwardOpConf);

}  // namespace oneflow
