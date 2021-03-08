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

void GenerateOptimizerOpConf(JobPassCtx* ctx, const OpNode& var_op_node,
                             const std::string& model_diff_lbn, const OptimizerConf optimizer_conf,
                             JobBuilder* job_builder) {
  const VariableOp* var_op = dynamic_cast<const VariableOp*>(&var_op_node.op());
  CHECK_NOTNULL(var_op);
  const std::string momentum_var_op_name = var_op->op_name() + "-momentum";
  OperatorConf momentum_var(var_op->op_conf());
  InitializerConf constant_initializer;
  constant_initializer.mutable_constant_conf()->set_value(0.f);
  *(momentum_var.mutable_variable_conf()->mutable_initializer()) = constant_initializer;
  momentum_var.set_name(momentum_var_op_name);
  momentum_var.mutable_variable_conf()->set_out("out");
  momentum_var.set_scope_symbol_id(var_op->op_conf().scope_symbol_id());
  job_builder->AddOps(var_op_node.parallel_desc().parallel_conf(), {momentum_var});

  user_op::UserOpConfWrapperBuilder lars_update_op_builder(var_op->op_name() + "_optimizer");
  lars_update_op_builder.OpTypeName("lars_update")
      .Input("model", GenLogicalBlobName(var_op->BnInOp2Lbi("out")))
      .Input("model_diff", model_diff_lbn)
      .Input("learning_rate", optimizer_conf.learning_rate_lbn())
      .Input("train_step", job_builder->job().job_conf().train_conf().train_step_lbn())
      .Input("momentum",
             GenLogicalBlobName(momentum_var_op_name, momentum_var.variable_conf().out()))
      .Attr<float>("momentum_beta", optimizer_conf.lars_conf().momentum_beta())
      .Attr<float>("epsilon", optimizer_conf.lars_conf().epsilon())
      .Attr<float>("lars_coefficient", optimizer_conf.lars_conf().lars_coefficient())
      .Attr<float>("weight_decay", GetOptimizerWeightDecayRate(optimizer_conf, *var_op))
      .ScopeSymbolId(var_op->op_conf().scope_symbol_id());
  SetDynamicLossScaleSkipIf(ctx, &lars_update_op_builder);
  user_op::UserOpConfWrapper lars_update_op = lars_update_op_builder.Build();
  job_builder->AddOps(var_op_node.parallel_desc().parallel_conf(), {lars_update_op.op_conf()});
}

}  // namespace

REGISTER_OPTIMIZER(OptimizerConf::kLarsConf, &GenerateOptimizerOpConf);

}  // namespace oneflow
