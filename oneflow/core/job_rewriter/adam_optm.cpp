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

void SetScalarShapeAndSbpConf(OperatorConf* op_conf) {
  op_conf->mutable_variable_conf()->mutable_shape()->clear_dim();
  op_conf->mutable_variable_conf()->mutable_shape()->add_dim(1);
  op_conf->mutable_variable_conf()->mutable_split_axis()->clear_value();
  CHECK_NE(op_conf->name(), std::string(""));
}

void GenerateOptimizerOpConf(const VariableOp& op, const ParallelConf& parallel_conf,
                             JobBuilder* job_builder, const LogicalBlobId& diff_lbi_of_var_out) {
  const auto& train_conf = job_builder->job().job_conf().train_conf();
  const NormalModelUpdateOpUserConf& model_update_conf = train_conf.model_update_conf();

  OperatorConf m_var(GenerateAdamHelperVariableOpConf(op, "m", 0.f));
  OperatorConf v_var(GenerateAdamHelperVariableOpConf(op, "v", 0.f));
  job_builder->AddOps(parallel_conf, {m_var, v_var});

  user_op::UserOpConfWrapperBuilder adam_update_op_builder(op.op_name() + "_optimizer");
  float beta1;
  float beta2;
  float epsilon;
  bool do_bias_correction;
  if (model_update_conf.has_adam_conf()) {
    const AdamModelUpdateConf& adam_conf = model_update_conf.adam_conf();
    beta1 = adam_conf.beta1();
    beta2 = adam_conf.beta2();
    epsilon = adam_conf.epsilon();
    do_bias_correction = adam_conf.do_bias_correction();
  } else if (model_update_conf.has_lazy_adam_conf()) {
    const LazyAdamModelUpdateConf& lazy_adam_conf = model_update_conf.lazy_adam_conf();
    beta1 = lazy_adam_conf.beta1();
    beta2 = lazy_adam_conf.beta2();
    epsilon = lazy_adam_conf.epsilon();
    do_bias_correction = true;
  } else {
    UNIMPLEMENTED();
  }
  adam_update_op_builder.OpTypeName("adam_update")
      .Input("model", GenLogicalBlobName(op.BnInOp2Lbi("out")))
      .Input("model_diff", GenLogicalBlobName(diff_lbi_of_var_out))
      .Input("learning_rate", train_conf.primary_lr_lbn())
      .Input("m", GenVariableOutputLbn(m_var))
      .Input("v", GenVariableOutputLbn(v_var))
      .Attr<float>("beta1", beta1)
      .Attr<float>("beta2", beta2)
      .Attr<float>("epsilon", epsilon)
      .Attr<bool>("do_bias_correction", do_bias_correction)
      .Attr<float>("weight_decay", GetOptimizerWeightDecayRate(model_update_conf, op))
      .ScopeSymbolId(op.op_conf().scope_symbol_id());

  if (do_bias_correction) {
    OperatorConf beta1_t_var;
    OperatorConf beta2_t_var;
    beta1_t_var = GenerateAdamHelperVariableOpConf(op, "beta1_t", beta1);
    SetScalarShapeAndSbpConf(&beta1_t_var);
    beta2_t_var = GenerateAdamHelperVariableOpConf(op, "beta2_t", beta2);
    SetScalarShapeAndSbpConf(&beta2_t_var);
    job_builder->AddOps(parallel_conf, {beta1_t_var, beta2_t_var});
    adam_update_op_builder.Input("beta1_t", GenVariableOutputLbn(beta1_t_var));
    adam_update_op_builder.Input("beta2_t", GenVariableOutputLbn(beta2_t_var));
  }

  const auto adam_update_op = adam_update_op_builder.Build();
  job_builder->AddOps(parallel_conf, {adam_update_op.op_conf()});
}

}  // namespace

REGISTER_OPTIMIZER(NormalModelUpdateOpUserConf::kAdamConf, &GenerateOptimizerOpConf);
REGISTER_OPTIMIZER(NormalModelUpdateOpUserConf::kLazyAdamConf, &GenerateOptimizerOpConf);

}  // namespace oneflow
