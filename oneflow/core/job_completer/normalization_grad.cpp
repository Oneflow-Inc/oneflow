#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  CHECK(op.op_conf().has_normalization_conf());
  const NormalizationOpConf& conf = op.op_conf().normalization_conf();
  LogicalBlobId* dx_lbi = DiffLbi4BnInOp("in");
  LogicalBlobId* gamma_diff_lbi = conf.has_gamma() ? DiffLbi4BnInOp("gamma") : nullptr;
  LogicalBlobId* beta_diff_lbi = conf.has_beta() ? DiffLbi4BnInOp("beta") : nullptr;
  CHECK(dx_lbi != nullptr || gamma_diff_lbi != nullptr || beta_diff_lbi != nullptr);
  OperatorConf normalization_grad_op;
  normalization_grad_op.set_name("System-AutoGrad-" + op.op_name());
  NormalizationGradOpConf* grad_conf = normalization_grad_op.mutable_normalization_grad_conf();
  grad_conf->set_axis(conf.axis());
  grad_conf->set_epsilon(conf.epsilon());
  grad_conf->set_dy(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
  grad_conf->set_x(GenLogicalBlobName(op.BnInOp2Lbi("in")));
  if (conf.is_training()) {
    grad_conf->set_mean(GenLogicalBlobName(op.BnInOp2Lbi("mean")));
    grad_conf->set_inv_variance(GenLogicalBlobName(op.BnInOp2Lbi("inv_variance")));
  } else {
    grad_conf->set_mean(GenLogicalBlobName(op.BnInOp2Lbi("moving_mean")));
    OperatorConf scarlar_add_op;
    scarlar_add_op.set_name("System-AutoGrad-" + op.op_name() + "-VarianceAddEpsilon");
    ScalarAddOpConf* scalar_add_conf = scarlar_add_op.mutable_scalar_add_conf();
    scalar_add_conf->set_in(GenLogicalBlobName(op.BnInOp2Lbi("moving_variance")));
    scalar_add_conf->set_float_operand(conf.epsilon());
    scalar_add_conf->set_out("out");
    OperatorConf rsqrt_op;
    rsqrt_op.set_name("System-AutoGrad-" + op.op_name() + "-InvVarianceRsqrt");
    RsqrtOpConf* rsqrt_conf = rsqrt_op.mutable_rsqrt_conf();
    rsqrt_conf->set_in(scalar_add_conf->out());
    rsqrt_conf->set_out("out");
    op_confs->push_back(rsqrt_op);
    LogicalBlobId inv_variance;
    inv_variance.set_op_name(rsqrt_op.name());
    inv_variance.set_blob_name(rsqrt_conf->out());
    grad_conf->set_inv_variance(GenLogicalBlobName(inv_variance));
  }
  if (conf.has_gamma()) { grad_conf->set_gamma(GenLogicalBlobName(op.BnInOp2Lbi("gamma"))); }
  if (dx_lbi != nullptr) {
    grad_conf->set_dx("dx");
    dx_lbi->set_op_name(normalization_grad_op.name());
    dx_lbi->set_blob_name(grad_conf->dx());
  }
  if (gamma_diff_lbi != nullptr) {
    grad_conf->set_gamma_diff("gamma_diff");
    gamma_diff_lbi->set_op_name(normalization_grad_op.name());
    gamma_diff_lbi->set_blob_name(grad_conf->gamma_diff());
  }
  if (beta_diff_lbi != nullptr) {
    grad_conf->set_beta_diff("beta_diff");
    beta_diff_lbi->set_op_name(normalization_grad_op.name());
    beta_diff_lbi->set_blob_name(grad_conf->beta_diff());
  }
  op_confs->emplace_back(normalization_grad_op);
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kNormalizationConf, &GenerateBackwardOpConf);

}  // namespace oneflow
