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

std::string GenVariableOutputLbn(const OperatorConf& op_conf) {
  CHECK(op_conf.has_variable_conf());
  return GenLogicalBlobName(op_conf.name(), op_conf.variable_conf().out());
}

OperatorConf GenerateAdamHelperVariableOpConf(const VariableOp& op, const std::string& name,
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

void GenerateOptimizerOpConf(JobPassCtx* ctx, const OpNode& var_op_node,
                             const std::string& model_diff_lbn, const OptimizerConf& optimizer_conf,
                             JobBuilder* job_builder) {
  const VariableOp* var_op = dynamic_cast<const VariableOp*>(&var_op_node.op());
  CHECK_NOTNULL(var_op);

  OperatorConf m_var(GenerateAdamHelperVariableOpConf(*var_op, "m", 0.f));
  OperatorConf v_var(GenerateAdamHelperVariableOpConf(*var_op, "v", 0.f));
  OperatorConf max_v_var(GenerateAdamHelperVariableOpConf(*var_op, "max_v", 0.f));

  job_builder->AddOps(var_op_node.parallel_desc().parallel_conf(), {m_var, v_var, max_v_var});

  user_op::UserOpConfWrapperBuilder adam_update_op_builder(var_op->op_name() + "_optimizer");
  float beta1;
  float beta2;
  float epsilon;
  bool do_bias_correction;
  bool amsgrad;
  if (optimizer_conf.has_adam_conf()) {
    const AdamModelUpdateConf& adam_conf = optimizer_conf.adam_conf();
    beta1 = adam_conf.beta1();
    beta2 = adam_conf.beta2();
    epsilon = adam_conf.epsilon();
    do_bias_correction = adam_conf.do_bias_correction();
    amsgrad = adam_conf.amsgrad();
  } else if (optimizer_conf.has_lazy_adam_conf()) {
    const LazyAdamModelUpdateConf& lazy_adam_conf = optimizer_conf.lazy_adam_conf();
    beta1 = lazy_adam_conf.beta1();
    beta2 = lazy_adam_conf.beta2();
    epsilon = lazy_adam_conf.epsilon();
    do_bias_correction = lazy_adam_conf.do_bias_correction();
    amsgrad = lazy_adam_conf.amsgrad();
  } else {
    UNIMPLEMENTED();
  }
  const std::string& train_step_lbn = job_builder->job().job_conf().train_conf().train_step_lbn();
  const std::string& learning_rate_lbn = optimizer_conf.learning_rate_lbn();

  adam_update_op_builder.OpTypeName("adam_update")
      .Input("model", GenLogicalBlobName(var_op->BnInOp2Lbi("out")))
      .Input("model_diff", model_diff_lbn)
      .Input("learning_rate", learning_rate_lbn)
      .Input("train_step", train_step_lbn)
      .Input("m", GenVariableOutputLbn(m_var))
      .Input("v", GenVariableOutputLbn(v_var))
      .Input("max_v", GenVariableOutputLbn(max_v_var))
      .Attr<float>("beta1", beta1)
      .Attr<float>("beta2", beta2)
      .Attr<float>("epsilon", epsilon)
      .Attr<float>("weight_decay", GetOptimizerWeightDecayRate(optimizer_conf, *var_op))
      .Attr<bool>("amsgrad", amsgrad)
      .Attr<bool>("do_bias_correction", do_bias_correction)
      .ScopeSymbolId(var_op->op_conf().scope_symbol_id());
  SetDynamicLossScaleSkipIf(ctx, &adam_update_op_builder);
  const auto adam_update_op = adam_update_op_builder.Build();
  job_builder->AddOps(var_op_node.parallel_desc().parallel_conf(), {adam_update_op.op_conf()});
}

}  // namespace

REGISTER_OPTIMIZER(OptimizerConf::kAdamConf, &GenerateOptimizerOpConf);
REGISTER_OPTIMIZER(OptimizerConf::kLazyAdamConf, &GenerateOptimizerOpConf);

}  // namespace oneflow
