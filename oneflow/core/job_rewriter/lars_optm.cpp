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

void GenerateOptimizerOpConf(JobPassCtx* ctx, const VariableOp& op,
                             const ParallelConf& parallel_conf, JobBuilder* job_builder,
                             const LogicalBlobId& diff_lbi_of_var_out) {
  OperatorConf momentum_var(op.op_conf());
  InitializerConf constant_initializer;
  constant_initializer.mutable_constant_conf()->set_value(0.f);
  *(momentum_var.mutable_variable_conf()->mutable_initializer()) = constant_initializer;
  momentum_var.set_name(op.op_name() + "-momentum");
  momentum_var.mutable_variable_conf()->set_out("out");
  momentum_var.set_scope_symbol_id(op.op_conf().scope_symbol_id());
  job_builder->AddOps(parallel_conf, {momentum_var});

  OperatorConf mdupdt_op;
  mdupdt_op.set_name(op.op_name() + "_optimizer");
  auto* mdupdt_op_conf = mdupdt_op.mutable_lars_model_update_conf();
  ConstructMdUpdtOpConf(op, diff_lbi_of_var_out, job_builder, mdupdt_op_conf);
  mdupdt_op_conf->set_momentum(momentum_var.name() + "/out");
  mdupdt_op.set_scope_symbol_id(op.op_conf().scope_symbol_id());
  job_builder->AddOps(parallel_conf, {mdupdt_op});
}

}  // namespace

REGISTER_OPTIMIZER(NormalModelUpdateOpUserConf::kLarsConf, &GenerateOptimizerOpConf);

}  // namespace oneflow
