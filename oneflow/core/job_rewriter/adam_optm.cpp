/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/job_rewriter/optimizer.h"

namespace oneflow {

namespace {

OperatorConf GenerateAdamHelperVariableOpConf(const VariableOp& op, const std::string& name,
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
  OperatorConf m_var(GenerateAdamHelperVariableOpConf(op, "m", 0.f));
  OperatorConf v_var(GenerateAdamHelperVariableOpConf(op, "v", 0.f));
  m_var.set_scope_symbol_id(op.op_conf().scope_symbol_id());
  v_var.set_scope_symbol_id(op.op_conf().scope_symbol_id());
  job_builder->AddOps(parallel_conf, {m_var, v_var});

  OperatorConf mdupdt_op;
  mdupdt_op.set_name(op.op_name() + "_optimizer");
  auto* mdupdt_op_conf = mdupdt_op.mutable_adam_model_update_conf();
  *(mdupdt_op_conf->mutable_user_conf()) =
      GlobalJobDesc().job_conf().train_conf().model_update_conf();
  OperatorConf beta1_t_var;
  OperatorConf beta2_t_var;
  const AdamModelUpdateConf& adam_conf = mdupdt_op_conf->user_conf().adam_conf();
  if (adam_conf.do_bias_correction()) {
    beta1_t_var = GenerateAdamHelperVariableOpConf(op, "beta1_t", adam_conf.beta1());
    SetScalarShapeAndSbpConf(&beta1_t_var);
    beta2_t_var = GenerateAdamHelperVariableOpConf(op, "beta2_t", adam_conf.beta2());
    SetScalarShapeAndSbpConf(&beta2_t_var);
    beta1_t_var.set_scope_symbol_id(op.op_conf().scope_symbol_id());
    beta2_t_var.set_scope_symbol_id(op.op_conf().scope_symbol_id());
    job_builder->AddOps(parallel_conf, {beta1_t_var, beta2_t_var});
  }
  ConstructMdUpdtOpConf(op, diff_lbi_of_var_out, job_builder, mdupdt_op_conf);
  mdupdt_op_conf->set_m(m_var.name() + "/out");
  mdupdt_op_conf->set_v(v_var.name() + "/out");
  if (adam_conf.do_bias_correction()) {
    mdupdt_op_conf->set_beta1_t(beta1_t_var.name() + "/out");
    mdupdt_op_conf->set_beta2_t(beta2_t_var.name() + "/out");
  }
  mdupdt_op.set_scope_symbol_id(op.op_conf().scope_symbol_id());
  job_builder->AddOps(parallel_conf, {mdupdt_op});
}

}  // namespace

REGISTER_OPTIMIZER(NormalModelUpdateOpUserConf::kAdamConf, &GenerateOptimizerOpConf);

}  // namespace oneflow
