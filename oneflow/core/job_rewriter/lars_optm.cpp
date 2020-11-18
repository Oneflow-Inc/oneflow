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
#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace {

void GenerateOptimizerOpConf(JobPassCtx* ctx, const VariableOp& op,
                             const ParallelConf& parallel_conf, JobBuilder* job_builder,
                             const LogicalBlobId& diff_lbi_of_var_out) {
  const auto& train_conf = job_builder->job().job_conf().train_conf();
  const NormalModelUpdateOpUserConf& model_update_conf = train_conf.model_update_conf();

  const std::string momentum_var_op_name = op.op_name() + "-momentum";
  OperatorConf momentum_var(op.op_conf());
  InitializerConf constant_initializer;
  constant_initializer.mutable_constant_conf()->set_value(0.f);
  *(momentum_var.mutable_variable_conf()->mutable_initializer()) = constant_initializer;
  momentum_var.set_name(momentum_var_op_name);
  momentum_var.mutable_variable_conf()->set_out("out");
  momentum_var.set_scope_symbol_id(op.op_conf().scope_symbol_id());
  job_builder->AddOps(parallel_conf, {momentum_var});

  user_op::UserOpConfWrapperBuilder lars_update_op_builder(op.op_name() + "_optimizer");
  lars_update_op_builder.OpTypeName("lars_update")
      .Input("model", GenLogicalBlobName(op.BnInOp2Lbi("out")))
      .Input("model_diff", GenLogicalBlobName(diff_lbi_of_var_out))
      .Input("learning_rate", train_conf.primary_lr_lbn())
      .Input("train_step", train_conf.train_step_lbn())
      .Input("momentum",
             GenLogicalBlobName(momentum_var_op_name, momentum_var.variable_conf().out()))
      .Attr<float>("momentum_beta", model_update_conf.lars_conf().momentum_beta())
      .Attr<float>("epsilon", model_update_conf.lars_conf().epsilon())
      .Attr<float>("lars_coefficient", model_update_conf.lars_conf().lars_coefficient())
      .Attr<float>("weight_decay", GetOptimizerWeightDecayRate(model_update_conf, op))
      .ScopeSymbolId(op.op_conf().scope_symbol_id());
  user_op::UserOpConfWrapper lars_update_op = lars_update_op_builder.Build();
  job_builder->AddOps(parallel_conf, {lars_update_op.op_conf()});
}

}  // namespace

REGISTER_OPTIMIZER(NormalModelUpdateOpUserConf::kLarsConf, &GenerateOptimizerOpConf);

}  // namespace oneflow
