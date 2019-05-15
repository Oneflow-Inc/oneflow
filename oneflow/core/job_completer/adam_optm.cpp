#include "oneflow/core/job_completer/optimizer.h"

namespace oneflow {

namespace {

OperatorConf GenerateAdamHelperVariableOpConf(const VariableOp& op, const std::string& name,
                                              JobBuilder* job_builder) {
  OperatorConf helper_variable_op(op.op_conf());
  helper_variable_op.set_name(op.op_name() + "-" + name);
  helper_variable_op.mutable_variable_conf()->set_out("out");
  BindTwoVariableOpObnSbpConf(helper_variable_op.name(), op.op_name(), job_builder);
  return helper_variable_op;
}

void SetScalarShapeAndSbpConf(OperatorConf* op_conf, JobBuilder* job_builder) {
  op_conf->mutable_variable_conf()->mutable_shape()->clear_dim();
  op_conf->mutable_variable_conf()->mutable_shape()->add_dim(1);
  CHECK_NE(op_conf->name(), std::string(""));
  job_builder->MutSbpParallel4Oba(GenOpBlobArg(op_conf->name(), "out"))
      ->mutable_broadcast_parallel();
}

void GenerateOptimizerOpConf(
    const VariableOp& op, const ParallelConf& parallel_conf, JobBuilder* job_builder,
    const std::function<const LogicalBlobId&(const std::string&)>& DiffLbi4BnInOp,
    const LogicalBlobId& total_loss_instance_num_lbi) {
  const OperatorConf& m_var = GenerateAdamHelperVariableOpConf(op, "m", job_builder);
  const OperatorConf& v_var = GenerateAdamHelperVariableOpConf(op, "v", job_builder);
  job_builder->AddOps(parallel_conf, {m_var, v_var});

  OperatorConf mdupdt_op;
  mdupdt_op.set_name(op.op_name() + "_optimizer");
  auto* mdupdt_op_conf = mdupdt_op.mutable_adam_model_update_conf();
  *(mdupdt_op_conf->mutable_user_conf()) = Global<JobDesc>::Get()
                                               ->other_conf()
                                               .predict_conf()
                                               .tmp_split_fw_bw_train_conf()
                                               .model_update_conf();
  OperatorConf beta1_t_var;
  OperatorConf beta2_t_var;
  if (mdupdt_op_conf->user_conf().adam_conf().do_bias_correction()) {
    beta1_t_var = GenerateAdamHelperVariableOpConf(op, "beta1_t", job_builder);
    beta2_t_var = GenerateAdamHelperVariableOpConf(op, "beta2_t", job_builder);
    job_builder->AddOps(parallel_conf, {beta1_t_var, beta2_t_var});
    SetScalarShapeAndSbpConf(&beta1_t_var, job_builder);
    SetScalarShapeAndSbpConf(&beta2_t_var, job_builder);
  }
  ConstructMdUpdtOpConf(op, DiffLbi4BnInOp, total_loss_instance_num_lbi, mdupdt_op_conf);
  mdupdt_op_conf->set_m(m_var.name() + "/out");
  mdupdt_op_conf->set_v(v_var.name() + "/out");
  if (mdupdt_op_conf->user_conf().adam_conf().do_bias_correction()) {
    mdupdt_op_conf->set_beta1_t(beta1_t_var.name() + "/out");
    mdupdt_op_conf->set_beta2_t(beta2_t_var.name() + "/out");
  }
  job_builder->AddOps(parallel_conf, {mdupdt_op});
}

}  // namespace

REGISTER_OPTIMIZER(NormalModelUpdateOpUserConf::kAdamConf, &GenerateOptimizerOpConf);

}  // namespace oneflow
