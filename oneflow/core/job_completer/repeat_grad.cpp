#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  CHECK(op.op_conf().has_repeat_conf());
  if (DiffLbi4BnInOp("in") != nullptr) {
    OperatorConf grad_op{};
    grad_op.set_name("System-AutoGrad-" + op.op_name());
    AccOpConf* acc_op_conf = grad_op.mutable_acc_conf();
    acc_op_conf->set_one(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    acc_op_conf->set_acc("acc");
    acc_op_conf->set_max_acc_num(op.op_conf().repeat_conf().repeat_num());
    op_confs->push_back(grad_op);
    DiffLbi4BnInOp("in")->set_op_name(grad_op.name());
    DiffLbi4BnInOp("in")->set_blob_name(acc_op_conf->acc());
  }
}

REGISTER_OP_GRAD(OperatorConf::kRepeatConf, &GenerateBackwardOpConf);

}  // namespace

}  // namespace oneflow
