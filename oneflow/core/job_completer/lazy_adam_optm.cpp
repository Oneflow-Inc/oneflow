#include "oneflow/core/job_completer/optimizer.h"

namespace oneflow {

namespace {

OperatorConf GenerateLazyAdamHelperVariableOpConf(const VariableOp& op, const std::string& name,
                                                  const float initial_value) {
  OperatorConf helper_variable_op(op.op_conf());
  helper_variable_op.set_name(op.op_name() + "-" + name);
  helper_variable_op.mutable_variable_conf()->set_out("out");
  InitializerConf constant_initializer;
  constant_initializer.mutable_constant_conf()->set_value(initial_value);
  *(helper_variable_op.mutable_variable_conf()->mutable_initializer()) = constant_initializer;
  return helper_variable_op;
}

void SetScalarShapeAndSbpConf(OperatorConf* op_conf) {
  op_conf->mutable_variable_conf()->mutable_shape()->clear_dim();
  op_conf->mutable_variable_conf()->mutable_shape()->add_dim(1);
  op_conf->mutable_variable_conf()->mutable_split_axis()->clear_value();
  CHECK_NE(op_conf->name(), std::string(""));
}

void GenerateOptimizerOpConf(const VariableOp& op, const ParallelConf& parallel_conf,
                             JobBuilder* job_builder, const LogicalBlobId& diff_lbi_of_var_out) {
  const OperatorConf& m_var = GenerateLazyAdamHelperVariableOpConf(op, "m", 0.f);
  const OperatorConf& v_var = GenerateLazyAdamHelperVariableOpConf(op, "v", 0.f);
  job_builder->AddOps(parallel_conf, {m_var, v_var});

  OperatorConf mdupdt_op;
  mdupdt_op.set_name(op.op_name() + "_optimizer");
  auto* mdupdt_op_conf = mdupdt_op.mutable_lazy_adam_model_update_conf();
  *(mdupdt_op_conf->mutable_user_conf()) =
      GlobalJobDesc().job_conf().train_conf().model_update_conf();
  OperatorConf beta1_t_var;
  OperatorConf beta2_t_var;
  const LazyAdamModelUpdateConf& lazy_adam_conf = mdupdt_op_conf->user_conf().lazy_adam_conf();
  beta1_t_var = GenerateLazyAdamHelperVariableOpConf(op, "beta1_t", lazy_adam_conf.beta1());
  SetScalarShapeAndSbpConf(&beta1_t_var);
  beta2_t_var = GenerateLazyAdamHelperVariableOpConf(op, "beta2_t", lazy_adam_conf.beta2());
  SetScalarShapeAndSbpConf(&beta2_t_var);
  job_builder->AddOps(parallel_conf, {beta1_t_var, beta2_t_var});
  ConstructMdUpdtOpConf(op, diff_lbi_of_var_out, job_builder, mdupdt_op_conf);
  mdupdt_op_conf->set_m(m_var.name() + "/out");
  mdupdt_op_conf->set_v(v_var.name() + "/out");
  mdupdt_op_conf->set_beta1_t(beta1_t_var.name() + "/out");
  mdupdt_op_conf->set_beta2_t(beta2_t_var.name() + "/out");
  job_builder->AddOps(parallel_conf, {mdupdt_op});
}

}  // namespace

REGISTER_OPTIMIZER(NormalModelUpdateOpUserConf::kLazyAdamConf, &GenerateOptimizerOpConf);

}  // namespace oneflow
