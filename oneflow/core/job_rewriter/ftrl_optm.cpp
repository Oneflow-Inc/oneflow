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

OperatorConf GenerateFtrlHelperVariableConf(const VariableOp& op, const std::string& name,
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

void GenerateFtrlOptimizerOpConf(JobPassCtx* ctx, const OpNode& var_op_node,
                                 const std::string& model_diff_lbn,
                                 const OptimizerConf& optimizer_conf, JobBuilder* job_builder) {
  const VariableOp* var_op = dynamic_cast<const VariableOp*>(&var_op_node.op());
  CHECK_NOTNULL(var_op);

  user_op::UserOpConfWrapperBuilder ftrl_update_op_builder(var_op->op_name() + "_optimizer");
  float lr_power = 0.0;
  float initial_accumulator_value = 0.0;
  float lambda1 = 0.0;
  float lambda2 = 0.0;
  float beta = 0.0;

  const FtrlModelUpdateConf& ftrl_conf = optimizer_conf.ftrl_conf();
  lr_power = ftrl_conf.lr_power();
  initial_accumulator_value = ftrl_conf.initial_accumulator_value();
  lambda1 = ftrl_conf.lambda1();
  lambda2 = ftrl_conf.lambda2();
  beta = ftrl_conf.beta();

  const std::string& learning_rate_lbn = optimizer_conf.learning_rate_lbn();
  OperatorConf accumulator_var(
      GenerateFtrlHelperVariableConf(*var_op, "accumulate", initial_accumulator_value));
  OperatorConf z_var(GenerateFtrlHelperVariableConf(*var_op, "z", 0.0));
  job_builder->AddOps(var_op_node.parallel_desc().parallel_conf(), {accumulator_var, z_var});

  ftrl_update_op_builder.OpTypeName("ftrl_update")
      .Input("model", GenLogicalBlobName(var_op->BnInOp2Lbi("out")))
      .Input("model_diff", model_diff_lbn)
      .Input("learning_rate", learning_rate_lbn)
      .Input("accumulate", GenVariableOutputLbn(accumulator_var))
      .Input("z", GenVariableOutputLbn(z_var))
      .Attr<float>("lr_power", lr_power)
      .Attr<float>("lambda1", lambda1)
      .Attr<float>("lambda2", lambda2)
      .Attr<float>("beta", beta)
      .Attr<float>("weight_decay", GetOptimizerWeightDecayRate(optimizer_conf, *var_op))
      .ScopeSymbolId(var_op->op_conf().scope_symbol_id());
  if (optimizer_conf.has_lr_scale()) {
    ftrl_update_op_builder.Attr<float>("learning_rate_scale", optimizer_conf.lr_scale());
  }
  SetDynamicLossScaleSkipIf(ctx, &ftrl_update_op_builder);
  const auto ftrl_update_op = ftrl_update_op_builder.Build();
  job_builder->AddOps(var_op_node.parallel_desc().parallel_conf(), {ftrl_update_op.op_conf()});
}

}  // namespace

REGISTER_OPTIMIZER(OptimizerConf::kFtrlConf, &GenerateFtrlOptimizerOpConf);

}  // namespace oneflow
