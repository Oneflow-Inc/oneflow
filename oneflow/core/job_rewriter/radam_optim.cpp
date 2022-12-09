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

namespace {

struct BiasCorrectionFactorCacheKey {
  float beta = 1.0;
  oneflow::ParallelConf parallel_conf;
};

bool operator==(const BiasCorrectionFactorCacheKey& lhs, const BiasCorrectionFactorCacheKey& rhs) {
  return (lhs.beta == rhs.beta) && (lhs.parallel_conf == rhs.parallel_conf);
}

}  // namespace

namespace std {

template<>
struct hash<BiasCorrectionFactorCacheKey> {
  size_t operator()(const BiasCorrectionFactorCacheKey& key) const {
    using namespace oneflow;
    return Hash(key.beta, key.parallel_conf);
  }
};

}  // namespace std

namespace oneflow {

class BiasCorrectionFactorState final : public JobPassState {
 public:
  BiasCorrectionFactorState() {}
  ~BiasCorrectionFactorState() override = default;

  std::string GetLbn(float beta, std::string bias_correction_name, ParallelConf parallel_conf,
                     const std::function<std::string(float beta_val, std::string op_name)>&
                         BiasCorrectionFactorStateOp) {
    BiasCorrectionFactorCacheKey cache_key;
    cache_key.beta = beta;
    cache_key.parallel_conf = parallel_conf;
    const auto& iter = key2lbn_.find(cache_key);
    if (iter != key2lbn_.end()) {
      return iter->second;
    } else {
      std::string lbn = BiasCorrectionFactorStateOp(beta, std::move(bias_correction_name));
      key2lbn_.emplace(cache_key, lbn);
      return lbn;
    }
  }

 private:
  HashMap<BiasCorrectionFactorCacheKey, std::string> key2lbn_;
};

namespace {

std::string GenVariableOutputLbn(const OperatorConf& op_conf) {
  CHECK(op_conf.has_variable_conf());
  return GenLogicalBlobName(op_conf.name(), op_conf.variable_conf().out());
}

OperatorConf GenerateRAdamHelperVariableOpConf(const VariableOp& op, const std::string& name,
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

  user_op::UserOpConfWrapperBuilder radam_update_op_builder(var_op->op_name() + "_optimizer");
  float beta1 = 0.9;
  float beta2 = 0.999;
  float rho_inf = 1999.0;
  float epsilon = 1e-8;
  bool do_bias_correction = true;
  if (optimizer_conf.has_radam_conf()) {
    const RAdamModelUpdateConf& radam_conf = optimizer_conf.radam_conf();
    beta1 = radam_conf.beta1();
    beta2 = radam_conf.beta2();
    rho_inf = radam_conf.rho_inf();
    epsilon = radam_conf.epsilon();
    do_bias_correction = radam_conf.do_bias_correction();
  } else {
    UNIMPLEMENTED();
  }
  OperatorConf m_var(GenerateRAdamHelperVariableOpConf(*var_op, "m", 0.f));
  OperatorConf v_var(GenerateRAdamHelperVariableOpConf(*var_op, "v", 0.f));
  job_builder->AddOps(var_op_node.parallel_desc().parallel_conf(), {m_var, v_var});

  const std::string& train_step_lbn = job_builder->job().job_conf().train_conf().train_step_lbn();
  const std::string& learning_rate_lbn = optimizer_conf.learning_rate_lbn();

  radam_update_op_builder.OpTypeName("radam_update")
      .Input("model", GenLogicalBlobName(var_op->BnInOp2Lbi("out")))
      .Input("model_diff", model_diff_lbn)
      .Input("learning_rate", learning_rate_lbn)
      .Input("m", GenVariableOutputLbn(m_var))
      .Input("v", GenVariableOutputLbn(v_var))
      .Attr<float>("beta1", beta1)
      .Attr<float>("beta2", beta2)
      .Attr<float>("rho_inf", rho_inf)
      .Attr<float>("epsilon", epsilon)
      .Attr<float>("weight_decay", GetOptimizerWeightDecayRate(optimizer_conf, *var_op))
      .Attr<bool>("do_bias_correction", do_bias_correction)
      .ScopeSymbolId(var_op->op_conf().scope_symbol_id());
  if (do_bias_correction) {
    const std::string& job_pass_state_key1 = "adam_bias_correction_factor";
    const bool has_state1 =
        CHECK_JUST(ctx->HasState<BiasCorrectionFactorState>(job_pass_state_key1));
    if (!has_state1) {
      CHECK_JUST(
          ctx->ResetState(job_pass_state_key1, std::make_unique<BiasCorrectionFactorState>()));
    }
    auto* state1 = CHECK_JUST(ctx->MutableState<BiasCorrectionFactorState>(job_pass_state_key1));

    const std::string& job_pass_state_key2 = "radam_bias_correction_factor";
    const bool has_state2 =
        CHECK_JUST(ctx->HasState<BiasCorrectionFactorState>(job_pass_state_key2));
    if (!has_state2) {
      CHECK_JUST(
          ctx->ResetState(job_pass_state_key2, std::make_unique<BiasCorrectionFactorState>()));
    }
    auto* state2 = CHECK_JUST(ctx->MutableState<BiasCorrectionFactorState>(job_pass_state_key2));

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
    auto AddAdamBiasCorrectionFactorOp = [&](float beta_val,
                                             const std::string& op_name) -> std::string {
      user_op::UserOpConfWrapperBuilder op_builder(var_op->op_name() + op_name);
      const auto adam_bias_correction_factor_op =
          op_builder.OpTypeName("adam_bias_correction_factor")
              .Input("train_step", train_step_lbn)
              .Attr<float>("beta", beta_val)
              .Output("out")
              .ScopeSymbolId(var_op->op_conf().scope_symbol_id())
              .Build();

      job_builder->AddOps(bias_correction_parallel_conf,
                          {adam_bias_correction_factor_op.op_conf()});
      return adam_bias_correction_factor_op.output("out", 0);
    };
    auto AddRAdamBiasCorrectionFactorOp = [&](float beta_val,
                                              const std::string& op_name) -> std::string {
      user_op::UserOpConfWrapperBuilder op_builder(var_op->op_name() + op_name);
      const auto radam_bias_correction_factor_op =
          op_builder.OpTypeName("radam_bias_correction_factor")
              .Input("train_step", train_step_lbn)
              .Attr<float>("beta", beta_val)
              .Output("out")
              .ScopeSymbolId(var_op->op_conf().scope_symbol_id())
              .Build();

      job_builder->AddOps(bias_correction_parallel_conf,
                          {radam_bias_correction_factor_op.op_conf()});
      return radam_bias_correction_factor_op.output("out", 0);
    };
    const std::string bias_correction1_lbn =
        state1->GetLbn(beta1, "radam_bias_correction_factor1", bias_correction_parallel_conf,
                       AddAdamBiasCorrectionFactorOp);
    const std::string bias_correction2_lbn =
        state1->GetLbn(beta2, "radam_bias_correction_factor2", bias_correction_parallel_conf,
                       AddAdamBiasCorrectionFactorOp);
    const std::string bias_correction2_numerator_lbn =
        state2->GetLbn(beta2, "radam_bias_correction_numerator_factor2",
                       bias_correction_parallel_conf, AddRAdamBiasCorrectionFactorOp);
    radam_update_op_builder.Input("bias_correction1", bias_correction1_lbn)
        .Input("bias_correction2", bias_correction2_lbn)
        .Input("bias_correction2_numerator", bias_correction2_numerator_lbn);
  }
  if (optimizer_conf.has_lr_scale()) {
    radam_update_op_builder.Attr<float>("learning_rate_scale", optimizer_conf.lr_scale());
  }

  SetDynamicLossScaleSkipIf(ctx, &radam_update_op_builder);
  const auto radam_update_op = radam_update_op_builder.Build();
  job_builder->AddOps(var_op_node.parallel_desc().parallel_conf(), {radam_update_op.op_conf()});
}

}  // namespace

REGISTER_OPTIMIZER(OptimizerConf::kRadamConf, &GenerateOptimizerOpConf);

}  // namespace oneflow
