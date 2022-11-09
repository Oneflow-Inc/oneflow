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
#include "oneflow/core/framework/user_op_conf.h"
#include "oneflow/core/job/initializer_conf.pb.h"
#include "oneflow/core/job/job_builder.h"
#include "oneflow/core/job/job_conf.pb.h"
#include "oneflow/core/job_rewriter/job_pass.h"
#include "oneflow/core/job_rewriter/optimizer.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/operator/variable_op.h"

namespace oneflow {

namespace {

std::string GenVariableOutputLbn(const OperatorConf& op_conf) {
  CHECK(op_conf.has_variable_conf());
  return GenLogicalBlobName(op_conf.name(), op_conf.variable_conf().out());
}

OperatorConf GenerateAdagradHelperVariableConf(const VariableOp& op, const std::string& name,
                                               const float initial_value) {
  OperatorConf helper_variable_op(op.op_conf());
  helper_variable_op.set_name(op.op_name() + "-" + name);
  helper_variable_op.mutable_variable_conf()->set_out("out");
  InitializerConf constant_initializer;
  constant_initializer.mutable_constant_conf()->set_value(initial_value);
  *(helper_variable_op.mutable_variable_conf()->mutable_initializer()) = constant_initializer;
  helper_variable_op.set_scope_symbol_id(op.op_conf().scope_symbol_id());
  return helper_variable_op;
}

void GenerateAdagradOptimizerOpConf(JobPassCtx* ctx, const OpNode& var_op_node,
                                    const std::string& model_diff_lbn,
                                    const OptimizerConf& optimizer_conf, JobBuilder* job_builder) {
  const VariableOp* var_op = dynamic_cast<const VariableOp*>(&var_op_node.op());
  CHECK_NOTNULL(var_op);

  user_op::UserOpConfWrapperBuilder adagrad_update_op_builder(var_op->op_name() + "_optimizer");
  float lr_decay = 0.0;
  float initial_accumulator_value = 0.0;
  float epsilon = 0.0;

  const AdagradModelUpdateConf& adagrad_conf = optimizer_conf.adagrad_conf();
  lr_decay = adagrad_conf.lr_decay();
  initial_accumulator_value = adagrad_conf.initial_accumulator_value();
  epsilon = adagrad_conf.epsilon();

  const std::string& train_step_lbn = job_builder->job().job_conf().train_conf().train_step_lbn();
  const std::string& learning_rate_lbn = optimizer_conf.learning_rate_lbn();

  OperatorConf sum_var(
      GenerateAdagradHelperVariableConf(*var_op, "sum", initial_accumulator_value));
  job_builder->AddOps(var_op_node.parallel_desc().parallel_conf(), {sum_var});

  adagrad_update_op_builder.OpTypeName("adagrad_update")
      .Input("model", GenLogicalBlobName(var_op->BnInOp2Lbi("out")))
      .Input("model_diff", model_diff_lbn)
      .Input("learning_rate", learning_rate_lbn)
      .Input("train_step", train_step_lbn)
      .Input("sum", GenVariableOutputLbn(sum_var))
      .Attr<float>("epsilon", epsilon)
      .Attr<float>("lr_decay", lr_decay)
      .Attr<float>("weight_decay", GetOptimizerWeightDecayRate(optimizer_conf, *var_op))
      .ScopeSymbolId(var_op->op_conf().scope_symbol_id());
  if (optimizer_conf.has_lr_scale()) {
    adagrad_update_op_builder.Attr<float>("learning_rate_scale", optimizer_conf.lr_scale());
  }
  SetDynamicLossScaleSkipIf(ctx, &adagrad_update_op_builder);
  const auto adagrad_update_op = adagrad_update_op_builder.Build();
  job_builder->AddOps(var_op_node.parallel_desc().parallel_conf(), {adagrad_update_op.op_conf()});
}

}  // namespace

REGISTER_OPTIMIZER(OptimizerConf::kAdagradConf, &GenerateAdagradOptimizerOpConf);

}  // namespace oneflow
