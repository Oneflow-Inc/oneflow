#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp,
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4BnInOp) {
  CHECK(op.op_conf().has_l2_normalize_conf());
  if (DiffLbi4BnInOp("in") != nullptr) {
    OperatorConf l2_normalize_grad_op;
    l2_normalize_grad_op.set_name("System-AutoGrad-" + op.op_name());
    L2NormalizeGradOpConf* conf = l2_normalize_grad_op.mutable_l2_normalize_grad_conf();
    conf->set_axis(op.op_conf().l2_normalize_conf().axis());
    conf->set_epsilon(op.op_conf().l2_normalize_conf().epsilon());
    conf->set_dy(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    conf->set_y(GenLogicalBlobName(op.BnInOp2Lbi("out")));
    conf->set_square_x_sum(GenLogicalBlobName(op.BnInOp2Lbi("square_x_sum")));
    conf->set_dx("dx");
    op_confs->push_back(l2_normalize_grad_op);
    DiffLbi4BnInOp("in")->set_op_name(l2_normalize_grad_op.name());
    DiffLbi4BnInOp("in")->set_blob_name(conf->dx());
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kL2NormalizeConf, &GenerateBackwardOpConf);

}  // namespace oneflow
