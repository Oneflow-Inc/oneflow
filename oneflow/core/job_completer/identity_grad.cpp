#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

void GenerateIdentityBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  CHECK(op.op_conf().has_identity_conf());
  if (DiffLbi4BnInOp("in") != nullptr) {
    OperatorConf grad_op{};
    grad_op.set_name("System-AutoGrad-" + op.op_name());
    IdentityOpConf* identity_op_conf = grad_op.mutable_identity_conf();
    identity_op_conf->set_in(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    identity_op_conf->set_out("out");
    op_confs->push_back(grad_op);
    DiffLbi4BnInOp("in")->set_op_name(grad_op.name());
    DiffLbi4BnInOp("in")->set_blob_name(identity_op_conf->out());
  }
}

REGISTER_OP_GRAD(OperatorConf::kIdentityConf, &GenerateIdentityBackwardOpConf);

}  // namespace

namespace {

void GenerateSleepBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  CHECK(op.op_conf().has_sleep_conf());
  if (DiffLbi4BnInOp("in") != nullptr) {
    OperatorConf grad_op{};
    grad_op.set_name("System-AutoGrad-" + op.op_name());
    SleepOpConf* sleep_op_conf = grad_op.mutable_sleep_conf();
    sleep_op_conf->set_in(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    sleep_op_conf->set_seconds(op.op_conf().sleep_conf().seconds());
    sleep_op_conf->set_out("out");
    op_confs->push_back(grad_op);
    DiffLbi4BnInOp("in")->set_op_name(grad_op.name());
    DiffLbi4BnInOp("in")->set_blob_name(sleep_op_conf->out());
  }
}

REGISTER_OP_GRAD(OperatorConf::kSleepConf, &GenerateSleepBackwardOpConf);

}  // namespace

}  // namespace oneflow
