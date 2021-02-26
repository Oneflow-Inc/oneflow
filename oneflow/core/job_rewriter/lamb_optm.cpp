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

OperatorConf GenerateLAMBHelperVariableOpConf(const VariableOp& op, const std::string& name,
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

void SetScalarShapeAndSbpConf(OperatorConf* op_conf) {
  op_conf->mutable_variable_conf()->mutable_shape()->clear_dim();
  op_conf->mutable_variable_conf()->mutable_shape()->add_dim(1);
  op_conf->mutable_variable_conf()->mutable_split_axis()->clear_value();
  CHECK_NE(op_conf->name(), std::string(""));
}

void GenerateOptimizerOpConf(JobPassCtx* ctx, const OpNode& var_op_node,
                             const std::string& model_diff_lbn, const OptimizerConf optimizer_conf,
                             JobBuilder* job_builder) {
  const VariableOp* var_op = dynamic_cast<const VariableOp*>(&var_op_node.op());
  CHECK_NOTNULL(var_op);
  OperatorConf m_var = GenerateLAMBHelperVariableOpConf(*var_op, "m", 0.f);
  OperatorConf v_var = GenerateLAMBHelperVariableOpConf(*var_op, "v", 0.f);
  job_builder->AddOps(var_op_node.parallel_desc().parallel_conf(), {m_var, v_var});

  OperatorConf beta1_t_var;
  OperatorConf beta2_t_var;
  const LambModelUpdateConf& lamb_conf = optimizer_conf.lamb_conf();
  beta1_t_var = GenerateLAMBHelperVariableOpConf(*var_op, "beta1_t", lamb_conf.beta1());
  SetScalarShapeAndSbpConf(&beta1_t_var);
  beta2_t_var = GenerateLAMBHelperVariableOpConf(*var_op, "beta2_t", lamb_conf.beta2());
  SetScalarShapeAndSbpConf(&beta2_t_var);
  job_builder->AddOps(var_op_node.parallel_desc().parallel_conf(), {beta1_t_var, beta2_t_var});

  user_op::UserOpConfWrapperBuilder lamb_update_op_builder(var_op->op_name() + "_optimizer");
  lamb_update_op_builder.OpTypeName("lamb_update")
      .Input("m", GenVariableOutputLbn(m_var))
      .Input("v", GenVariableOutputLbn(v_var))
      .Input("beta1_t", GenVariableOutputLbn(beta1_t_var))
      .Input("beta2_t", GenVariableOutputLbn(beta2_t_var))
      .Input("model", GenLogicalBlobName(var_op->BnInOp2Lbi("out")))
      .Input("model_diff", model_diff_lbn)
      .Input("learning_rate", optimizer_conf.learning_rate_lbn())
      .Attr<float>("beta1", lamb_conf.beta1())
      .Attr<float>("beta2", lamb_conf.beta2())
      .Attr<float>("epsilon", lamb_conf.epsilon())
      .Attr<float>("weight_decay", GetOptimizerWeightDecayRate(optimizer_conf, *var_op))
      .ScopeSymbolId(var_op->op_conf().scope_symbol_id());
  SetDynamicLossScaleSkipIf(ctx, &lamb_update_op_builder);
  const auto lamb_update_op = lamb_update_op_builder.Build();
  job_builder->AddOps(var_op_node.parallel_desc().parallel_conf(), {lamb_update_op.op_conf()});
}

}  // namespace

REGISTER_OPTIMIZER(OptimizerConf::kLambConf, &GenerateOptimizerOpConf);

}  // namespace oneflow
