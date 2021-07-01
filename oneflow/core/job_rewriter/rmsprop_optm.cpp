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

OperatorConf GenerateRmspropHelperVariableOpConf(const VariableOp& op, const std::string& name,
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
                             const std::string& model_diff_lbn, const OptimizerConf optimizer_conf,
                             JobBuilder* job_builder) {
  const VariableOp* var_op = dynamic_cast<const VariableOp*>(&var_op_node.op());
  CHECK_NOTNULL(var_op);
  OperatorConf mean_square_var(GenerateRmspropHelperVariableOpConf(*var_op, "mean_square", 0.f));
  job_builder->AddOps(var_op_node.parallel_desc().parallel_conf(), {mean_square_var});

  user_op::UserOpConfWrapperBuilder rmsprop_update_op_builder(var_op->op_name() + "_optimizer");
  const RMSPropModelUpdateConf& rmsprop_conf = optimizer_conf.rmsprop_conf();
  bool centered = rmsprop_conf.centered();
  rmsprop_update_op_builder.OpTypeName("rmsprop_update")
      .Input("model", GenLogicalBlobName(var_op->BnInOp2Lbi("out")))
      .Input("model_diff", model_diff_lbn)
      .Input("learning_rate", optimizer_conf.learning_rate_lbn())
      .Input("mean_square", GenVariableOutputLbn(mean_square_var))
      .Attr<bool>("centered", centered)
      .Attr<float>("epsilon", rmsprop_conf.epsilon())
      .Attr<float>("decay_rate", rmsprop_conf.decay_rate())
      .Attr<float>("weight_decay", GetOptimizerWeightDecayRate(optimizer_conf, *var_op))
      .ScopeSymbolId(var_op->op_conf().scope_symbol_id());
  SetDynamicLossScaleSkipIf(ctx, &rmsprop_update_op_builder);

  if (centered) {
    OperatorConf mean_gradient_var(
        GenerateRmspropHelperVariableOpConf(*var_op, "mean_gradient", 0.f));
    job_builder->AddOps(var_op_node.parallel_desc().parallel_conf(), {mean_gradient_var});
    rmsprop_update_op_builder.Input("mean_gradient", GenVariableOutputLbn(mean_gradient_var));
  }

  user_op::UserOpConfWrapper rmsprop_update_op = rmsprop_update_op_builder.Build();
  job_builder->AddOps(var_op_node.parallel_desc().parallel_conf(), {rmsprop_update_op.op_conf()});
}

}  // namespace

REGISTER_OPTIMIZER(OptimizerConf::kRmspropConf, &GenerateOptimizerOpConf);

}  // namespace oneflow
