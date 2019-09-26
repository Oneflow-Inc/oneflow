#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  CHECK(op.op_conf().has_smooth_l1_conf());
  if (DiffLbi4BnInOp("prediction") != nullptr) {
    auto smooth_l1_op_conf = op.op_conf().smooth_l1_conf();
    OperatorConf smooth_l1_grad_op;
    smooth_l1_grad_op.set_name(op.op_name() + "_grad");
    SmoothL1GradOpConf* smooth_l1_grad_op_conf = smooth_l1_grad_op.mutable_smooth_l1_grad_conf();
    smooth_l1_grad_op_conf->set_x(GenLogicalBlobName(op.BnInOp2Lbi("prediction")));
    smooth_l1_grad_op_conf->set_label(GenLogicalBlobName(op.BnInOp2Lbi("label")));
    smooth_l1_grad_op_conf->set_dy(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    smooth_l1_grad_op_conf->set_dx("dx");
    smooth_l1_grad_op_conf->set_beta(smooth_l1_op_conf.beta());
    smooth_l1_grad_op_conf->set_scale(smooth_l1_op_conf.scale());
    op_confs->push_back(smooth_l1_grad_op);
    DiffLbi4BnInOp("prediction")->set_op_name(smooth_l1_grad_op.name());
    DiffLbi4BnInOp("prediction")->set_blob_name("dx");
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kSmoothL1Conf, &GenerateBackwardOpConf);

}  // namespace oneflow
