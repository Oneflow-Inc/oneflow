#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  CHECK(op.op_conf().has_acc_conf());
  if (DiffLbi4BnInOp("one") != nullptr) {
    OperatorConf grad_op{};
    grad_op.set_name("System-AutoGrad-" + op.op_name());
    RepeatOpConf* repeat_op_conf = grad_op.mutable_repeat_conf();
    repeat_op_conf->set_in(GenLogicalBlobName(*DiffLbi4BnInOp("acc")));
    repeat_op_conf->set_out("out");
    repeat_op_conf->set_repeat_num(op.op_conf().acc_conf().max_acc_num());
    op_confs->push_back(grad_op);
    DiffLbi4BnInOp("one")->set_op_name(grad_op.name());
    DiffLbi4BnInOp("one")->set_blob_name(repeat_op_conf->out());
  }
}

REGISTER_OP_GRAD(OperatorConf::kAccConf, &GenerateBackwardOpConf);

}  // namespace

}  // namespace oneflow
