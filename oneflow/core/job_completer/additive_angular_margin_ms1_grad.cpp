#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  CHECK(op.op_conf().has_additive_angular_margin_ms1_conf());
  if (DiffLbi4BnInOp("in") != nullptr) {
    OperatorConf additive_angular_margin_ms1_grad_op;
    additive_angular_margin_ms1_grad_op.set_name(op.op_name() + "_grad");
    AdditiveAngularMarginMs1GradOpConf* conf =
        additive_angular_margin_ms1_grad_op.mutable_additive_angular_margin_ms1_grad_conf();
    conf->set_depth(op.op_conf().additive_angular_margin_ms1_conf().depth());
    conf->set_margin(op.op_conf().additive_angular_margin_ms1_conf().margin());
    conf->set_dy(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    conf->set_label(GenLogicalBlobName(op.BnInOp2Lbi("label")));
    conf->set_sin_theta_data(GenLogicalBlobName(op.BnInOp2Lbi("sin_theta_data")));
    conf->set_dx("dx");
    op_confs->push_back(additive_angular_margin_ms1_grad_op);
    DiffLbi4BnInOp("in")->set_op_name(additive_angular_margin_ms1_grad_op.name());
    DiffLbi4BnInOp("in")->set_blob_name(conf->dx());
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kAdditiveAngularMarginMs1Conf, &GenerateBackwardOpConf);

}  // namespace oneflow
