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

void GenerateOptimizerOpConf(const VariableOp& op, const ParallelConf& parallel_conf,
                             JobBuilder* job_builder, const LogicalBlobId& diff_lbi_of_var_out) {
  OperatorConf mean_square_var(GenerateAdamHelperVariableOpConf(op, "mean_square", 0.f));
  mean_square_var.set_scope_symbol_id(op.op_conf().scope_symbol_id());

  OperatorConf mdupdt_op;
  mdupdt_op.set_name(op.op_name() + "_optimizer");
  auto* mdupdt_op_conf = mdupdt_op.mutable_rmsprop_model_update_conf();
  *(mdupdt_op_conf->mutable_user_conf()) =
      GlobalJobDesc().job_conf().train_conf().model_update_conf();
  ConstructMdUpdtOpConf(op, diff_lbi_of_var_out, job_builder, mdupdt_op_conf);
  mdupdt_op_conf->set_mean_square(mean_square_var.name() + "/out");
  mdupdt_op.set_scope_symbol_id(op.op_conf().scope_symbol_id());
  if (GlobalJobDesc().job_conf().train_conf().model_update_conf().rmsprop_conf().centered()) {
    OperatorConf mean_gradient_var(GenerateAdamHelperVariableOpConf(op, "mean_gradient", 0.f));
    mean_gradient_var.set_scope_symbol_id(op.op_conf().scope_symbol_id());
    mdupdt_op_conf->set_mean_gradient(mean_gradient_var.name() + "/out");
    job_builder->AddOps(parallel_conf, {mean_square_var, mean_gradient_var, mdupdt_op});
  } else {
    job_builder->AddOps(parallel_conf, {mean_square_var, mdupdt_op});
  }
}

}  // namespace

REGISTER_OPTIMIZER(NormalModelUpdateOpUserConf::kRmspropConf, &GenerateOptimizerOpConf);

}  // namespace oneflow
