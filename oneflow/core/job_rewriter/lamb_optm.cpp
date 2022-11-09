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

struct BiasCorrectionFactorCacheKey {
  float beta = 1.0;
  ParallelConf parallel_conf;
};

bool operator==(const BiasCorrectionFactorCacheKey& lhs, const BiasCorrectionFactorCacheKey& rhs);

}  // namespace oneflow

namespace std {
template<>
struct hash<oneflow::BiasCorrectionFactorCacheKey> {
  size_t operator()(const oneflow::BiasCorrectionFactorCacheKey& key) const {
    using namespace oneflow;
    return Hash(key.beta, key.parallel_conf);
  }
};

}  // namespace std

namespace oneflow {

// Forward declaration for bias correction factor
class BiasCorrectionFactorState final : public JobPassState {
 public:
  BiasCorrectionFactorState() {}
  ~BiasCorrectionFactorState() override = default;

  std::string GetLbn(float beta, std::string bias_correction_name, ParallelConf parallel_conf,
                     const std::function<std::string(float beta_val, std::string op_name)>&
                         BiasCorrectionFactorStateOp);

 private:
  HashMap<BiasCorrectionFactorCacheKey, std::string> key2lbn_;
};

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

void SetScalarShapeAndNdSbpConf(const ParallelDesc& parallel_desc, OperatorConf* op_conf) {
  op_conf->mutable_variable_conf()->mutable_shape()->clear_dim();
  op_conf->mutable_variable_conf()->mutable_shape()->add_dim(1);
  op_conf->mutable_variable_conf()->clear_nd_sbp();
  FOR_RANGE(int, i, 0, parallel_desc.hierarchy()->NumAxes()) {
    *op_conf->mutable_variable_conf()->add_nd_sbp() = "B";
  }
  CHECK_NE(op_conf->name(), std::string(""));
}

void GenerateOptimizerOpConf(JobPassCtx* ctx, const OpNode& var_op_node,
                             const std::string& model_diff_lbn, const OptimizerConf& optimizer_conf,
                             JobBuilder* job_builder) {
  const VariableOp* var_op = dynamic_cast<const VariableOp*>(&var_op_node.op());
  CHECK_NOTNULL(var_op);

  OperatorConf m_var = GenerateLAMBHelperVariableOpConf(*var_op, "m", 0.f);
  OperatorConf v_var = GenerateLAMBHelperVariableOpConf(*var_op, "v", 0.f);

  job_builder->AddOps(var_op_node.parallel_desc().parallel_conf(), {m_var, v_var});

  user_op::UserOpConfWrapperBuilder lamb_update_op_builder(var_op->op_name() + "_optimizer");

  const LambModelUpdateConf& lamb_conf = optimizer_conf.lamb_conf();
  float beta1 = lamb_conf.beta1();
  float beta2 = lamb_conf.beta2();
  float epsilon = lamb_conf.epsilon();
  bool do_bias_correction = lamb_conf.do_bias_correction();

  const std::string& train_step_lbn = job_builder->job().job_conf().train_conf().train_step_lbn();
  const std::string& learning_rate_lbn = optimizer_conf.learning_rate_lbn();

  if (do_bias_correction) {
    // Reuse adam bias_correction job pass
    const std::string& job_pass_state_key = "adam_bias_correction_factor";
    const bool has_state = CHECK_JUST(ctx->HasState<BiasCorrectionFactorState>(job_pass_state_key));
    if (!has_state) {
      CHECK_JUST(
          ctx->ResetState(job_pass_state_key, std::make_unique<BiasCorrectionFactorState>()));
    }
    auto* state = CHECK_JUST(ctx->MutableState<BiasCorrectionFactorState>(job_pass_state_key));
    ParallelConf bias_correction_parallel_conf;
    const auto& lr_parallel_conf =
        CHECK_JUST(job_builder->ParallelConf4Lbi(GenLogicalBlobId(learning_rate_lbn)));
    const auto& train_step_parallel_conf =
        CHECK_JUST(job_builder->ParallelConf4Lbi(GenLogicalBlobId(train_step_lbn)));
    if (lr_parallel_conf == train_step_parallel_conf) {
      bias_correction_parallel_conf = lr_parallel_conf;
    } else {
      bias_correction_parallel_conf = var_op_node.parallel_desc().parallel_conf();
    }
    auto AddLambBiasCorrectionFactorOp = [&](float beta_val,
                                             const std::string& op_name) -> std::string {
      user_op::UserOpConfWrapperBuilder op_builder(var_op->op_name() + op_name);
      const auto lamb_bias_correction_factor_op =
          op_builder.OpTypeName("adam_bias_correction_factor")
              .Input("train_step", train_step_lbn)
              .Attr<float>("beta", beta_val)
              .Output("out")
              .ScopeSymbolId(var_op->op_conf().scope_symbol_id())
              .Build();

      job_builder->AddOps(bias_correction_parallel_conf,
                          {lamb_bias_correction_factor_op.op_conf()});
      return lamb_bias_correction_factor_op.output("out", 0);
    };

    const std::string bias_correction1_lbn =
        state->GetLbn(beta1, "lamb_bias_correction_factor1", bias_correction_parallel_conf,
                      AddLambBiasCorrectionFactorOp);
    const std::string bias_correction2_lbn =
        state->GetLbn(beta2, "lamb_bias_correction_factor2", bias_correction_parallel_conf,
                      AddLambBiasCorrectionFactorOp);

    lamb_update_op_builder.OpTypeName("lamb_update")
        .Input("model", GenLogicalBlobName(var_op->BnInOp2Lbi("out")))
        .Input("model_diff", model_diff_lbn)
        .Input("m", GenVariableOutputLbn(m_var))
        .Input("v", GenVariableOutputLbn(v_var))
        .Input("learning_rate", learning_rate_lbn)
        .Input("bias_correction1", bias_correction1_lbn)
        .Input("bias_correction2", bias_correction2_lbn)
        .Attr<float>("beta1", beta1)
        .Attr<float>("beta2", beta2)
        .Attr<float>("epsilon", epsilon)
        .Attr<float>("weight_decay", GetOptimizerWeightDecayRate(optimizer_conf, *var_op))
        .Attr<bool>("do_bias_correction", true)
        .ScopeSymbolId(var_op->op_conf().scope_symbol_id());
  } else {
    lamb_update_op_builder.OpTypeName("lamb_update")
        .Input("model", GenLogicalBlobName(var_op->BnInOp2Lbi("out")))
        .Input("model_diff", model_diff_lbn)
        .Input("m", GenVariableOutputLbn(m_var))
        .Input("v", GenVariableOutputLbn(v_var))
        .Input("learning_rate", learning_rate_lbn)
        .Attr<float>("beta1", beta1)
        .Attr<float>("beta2", beta2)
        .Attr<float>("epsilon", epsilon)
        .Attr<float>("weight_decay", GetOptimizerWeightDecayRate(optimizer_conf, *var_op))
        .Attr<bool>("do_bias_correction", false)
        .ScopeSymbolId(var_op->op_conf().scope_symbol_id());
  }

  if (optimizer_conf.has_lr_scale()) {
    lamb_update_op_builder.Attr<float>("learning_rate_scale", optimizer_conf.lr_scale());
  }

  SetDynamicLossScaleSkipIf(ctx, &lamb_update_op_builder);
  const auto lamb_update_op = lamb_update_op_builder.Build();
  job_builder->AddOps(var_op_node.parallel_desc().parallel_conf(), {lamb_update_op.op_conf()});
}

}  // namespace

REGISTER_OPTIMIZER(OptimizerConf::kLambConf, &GenerateOptimizerOpConf);

}  // namespace oneflow
