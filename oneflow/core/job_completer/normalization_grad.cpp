#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

void GenBnInDiffOpConfWhenTrainingFalse(
    const Operator& op, const OperatorConf& inv_variance_op_conf,
    std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp,
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4BnInOp) {
  OperatorConf reshape_gamma_op;
  reshape_gamma_op.set_name("System-AutoGrad-" + op.op_name() + "-ReshapeGamma");
  ReshapeOpConf* reshape_gamma_op_conf = reshape_gamma_op.mutable_reshape_conf();
  reshape_gamma_op_conf->set_out("out");
  reshape_gamma_op_conf->set_in(GenLogicalBlobName(op.BnInOp2Lbi("gamma")));
  const int32_t axis = op.op_conf().normalization_conf().axis();
  FOR_RANGE(size_t, i, 0, LogicalBlobDesc4BnInOp("in").shape().NumAxes()) {
    if (i != axis) {
      reshape_gamma_op_conf->mutable_shape()->add_dim(1);
    } else {
      reshape_gamma_op_conf->mutable_shape()->add_dim(
          LogicalBlobDesc4BnInOp("in").shape().At(axis));
    }
  }
  op_confs->push_back(reshape_gamma_op);

  OperatorConf reshape_inv_var_op;
  reshape_inv_var_op.set_name("System-AutoGrad-" + op.op_name() + "-ReshapeInvVar");
  ReshapeLikeOpConf* reshape_inv_var_op_conf = reshape_inv_var_op.mutable_reshape_like_conf();
  reshape_inv_var_op_conf->set_x(inv_variance_op_conf.name() + "/out");
  reshape_inv_var_op_conf->set_like(reshape_gamma_op.name() + "/out");
  reshape_inv_var_op_conf->set_y("y");
  op_confs->push_back(reshape_inv_var_op);
  OperatorConf broadcast_mul_gamma_op;
  broadcast_mul_gamma_op.set_name("System-AutoGrad-" + op.op_name() + "-BroadcastMulGamma");
  BroadcastMulOpConf* broadcast_mul_gamma_op_conf =
      broadcast_mul_gamma_op.mutable_broadcast_mul_conf();
  broadcast_mul_gamma_op_conf->set_a(reshape_gamma_op.name() + "/out");
  broadcast_mul_gamma_op_conf->set_b(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
  broadcast_mul_gamma_op_conf->set_out("out");
  op_confs->push_back(broadcast_mul_gamma_op);
  OperatorConf broadcast_mul_inv_var_op;
  broadcast_mul_inv_var_op.set_name("System-AutoGrad-" + op.op_name() + "-BroadcastMulInvVar");
  BroadcastMulOpConf* broadcast_mul_inv_var_op_conf =
      broadcast_mul_inv_var_op.mutable_broadcast_mul_conf();
  broadcast_mul_inv_var_op_conf->set_a(broadcast_mul_gamma_op.name() + "/out");
  broadcast_mul_inv_var_op_conf->set_b(reshape_inv_var_op.name() + "/y");
  broadcast_mul_inv_var_op_conf->set_out("out");
  op_confs->push_back(broadcast_mul_inv_var_op);
  DiffLbi4BnInOp("in")->set_op_name(broadcast_mul_inv_var_op.name());
  DiffLbi4BnInOp("in")->set_blob_name("out");
}

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp,
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4BnInOp) {
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
    if (dx_lbi != nullptr) {
      grad_conf->set_dx("dx");
      dx_lbi->set_op_name(normalization_grad_op.name());
      dx_lbi->set_blob_name(grad_conf->dx());
    }
  } else {
    grad_conf->set_mean(GenLogicalBlobName(op.BnInOp2Lbi("moving_mean")));
    OperatorConf rsqrt_op;
    rsqrt_op.set_name("System-AutoGrad-" + op.op_name() + "-InvVarianceRsqrt");
    RsqrtOpConf* rsqrt_conf = rsqrt_op.mutable_rsqrt_conf();
    rsqrt_conf->set_in(GenLogicalBlobName(op.BnInOp2Lbi("moving_variance")));
    rsqrt_conf->set_out("out");
    rsqrt_conf->set_epsilon(conf.epsilon());
    op_confs->push_back(rsqrt_op);
    LogicalBlobId inv_variance;
    inv_variance.set_op_name(rsqrt_op.name());
    inv_variance.set_blob_name(rsqrt_conf->out());
    grad_conf->set_inv_variance(GenLogicalBlobName(inv_variance));
    if (dx_lbi != nullptr) {
      GenBnInDiffOpConfWhenTrainingFalse(op, rsqrt_op, op_confs, DiffLbi4BnInOp,
                                         LogicalBlobDesc4BnInOp);
    }
  }
  if (conf.has_gamma()) { grad_conf->set_gamma(GenLogicalBlobName(op.BnInOp2Lbi("gamma"))); }
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
