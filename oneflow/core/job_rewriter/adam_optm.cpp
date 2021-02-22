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

struct AdamBiasCorrectionLearningRateCacheKey {
  float beta1;
  float beta2;
  std::string lr_lbn;
  std::string step_lbn;
  ParallelConf parallel_conf;
};

bool operator==(const AdamBiasCorrectionLearningRateCacheKey& lhs,
                const AdamBiasCorrectionLearningRateCacheKey& rhs) {
  return (lhs.beta1 == rhs.beta1) && (lhs.beta2 == rhs.beta2) && (lhs.lr_lbn == rhs.lr_lbn)
         && (lhs.step_lbn == rhs.step_lbn) && (lhs.parallel_conf == rhs.parallel_conf);
}

}  // namespace oneflow

namespace std {

template<>
struct hash<oneflow::AdamBiasCorrectionLearningRateCacheKey> {
  size_t operator()(const oneflow::AdamBiasCorrectionLearningRateCacheKey& key) const {
    const auto& str_hash = std::hash<std::string>();
    const auto& float_hash = std::hash<float>();
    const auto& parallel_conf_hash = std::hash<oneflow::ParallelConf>();
    return float_hash(key.beta1) ^ float_hash(key.beta2) ^ str_hash(key.lr_lbn)
           ^ str_hash(key.step_lbn) ^ parallel_conf_hash(key.parallel_conf);
  }
};

}  // namespace std

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

class AdamBiasCorrectionLearningRateState final : public JobPassState {
 public:
  AdamBiasCorrectionLearningRateState() {}
  ~AdamBiasCorrectionLearningRateState() override = default;

  std::string GetLbn(float beta1, float beta2, std::string lr_lbn, std::string step_lbn,
                     ParallelConf parallel_conf,
                     std::function<std::string()> AddAdamBiasCorrectionLearningRateOp) {
    AdamBiasCorrectionLearningRateCacheKey cache_key;
    cache_key.beta1 = beta1;
    cache_key.beta2 = beta2;
    cache_key.lr_lbn = lr_lbn;
    cache_key.step_lbn = step_lbn;
    cache_key.parallel_conf = parallel_conf;
    const auto& iter = key2lbn_.find(cache_key);
    if (iter != key2lbn_.end()) {
      return iter->second;
    } else {
      std::string lbn = AddAdamBiasCorrectionLearningRateOp();
      key2lbn_.emplace(cache_key, lbn);
      return lbn;
    }
  }

 private:
  HashMap<AdamBiasCorrectionLearningRateCacheKey, std::string> key2lbn_;
};

void GenerateOptimizerOpConf(JobPassCtx* ctx, const OpNode& var_op_node,
                             const std::string& model_diff_lbn, const OptimizerConf& optimizer_conf,
                             JobBuilder* job_builder) {
  const VariableOp* var_op = dynamic_cast<const VariableOp*>(&var_op_node.op());
  CHECK_NOTNULL(var_op);

  OperatorConf m_var(GenerateAdamHelperVariableOpConf(*var_op, "m", 0.f));
  OperatorConf v_var(GenerateAdamHelperVariableOpConf(*var_op, "v", 0.f));
  job_builder->AddOps(var_op_node.parallel_desc().parallel_conf(), {m_var, v_var});

  user_op::UserOpConfWrapperBuilder adam_update_op_builder(var_op->op_name() + "_optimizer");
  float beta1;
  float beta2;
  float epsilon;
  bool do_bias_correction;
  if (optimizer_conf.has_adam_conf()) {
    const AdamModelUpdateConf& adam_conf = optimizer_conf.adam_conf();
    beta1 = adam_conf.beta1();
    beta2 = adam_conf.beta2();
    epsilon = adam_conf.epsilon();
    do_bias_correction = adam_conf.do_bias_correction();
  } else if (optimizer_conf.has_lazy_adam_conf()) {
    const LazyAdamModelUpdateConf& lazy_adam_conf = optimizer_conf.lazy_adam_conf();
    beta1 = lazy_adam_conf.beta1();
    beta2 = lazy_adam_conf.beta2();
    epsilon = lazy_adam_conf.epsilon();
    do_bias_correction = true;
  } else {
    UNIMPLEMENTED();
  }
  const std::string& train_step_lbn = job_builder->job().job_conf().train_conf().train_step_lbn();
  const std::string& learning_rate_lbn = optimizer_conf.learning_rate_lbn();
  std::string lr_lbn;
  if (do_bias_correction) {
    const std::string& job_pass_state_key = "adam_bias_correction_learning_rate";
    const bool has_state =
        CHECK_JUST(ctx->HasState<AdamBiasCorrectionLearningRateState>(job_pass_state_key));
    if (!has_state) {
      ctx->ResetState(job_pass_state_key, std::make_unique<AdamBiasCorrectionLearningRateState>());
    }
    auto* state =
        CHECK_JUST(ctx->MutableState<AdamBiasCorrectionLearningRateState>(job_pass_state_key));
    ParallelConf bias_correction_parallel_conf;
    const auto& lr_parallel_conf =
        job_builder->ParallelConf4Lbi(GenLogicalBlobId(learning_rate_lbn));
    const auto& train_step_parallel_conf =
        job_builder->ParallelConf4Lbi(GenLogicalBlobId(train_step_lbn));
    if (lr_parallel_conf == train_step_parallel_conf) {
      bias_correction_parallel_conf = lr_parallel_conf;
    } else {
      bias_correction_parallel_conf = var_op_node.parallel_desc().parallel_conf();
    }
    auto AddAdamBiasCorrectionLearningRateOp = [&]() -> std::string {
      user_op::UserOpConfWrapperBuilder op_builder(var_op->op_name()
                                                   + "_adam_bias_correction_learning_rate");
      const auto adam_bias_correction_learning_rate_op =
          op_builder.OpTypeName("adam_bias_correction_learning_rate")
              .Input("learning_rate", learning_rate_lbn)
              .Input("train_step", train_step_lbn)
              .Attr<float>("beta1", beta1)
              .Attr<float>("beta2", beta2)
              .Output("out")
              .ScopeSymbolId(var_op->op_conf().scope_symbol_id())
              .Build();
      job_builder->AddOps(bias_correction_parallel_conf,
                          {adam_bias_correction_learning_rate_op.op_conf()});
      return adam_bias_correction_learning_rate_op.output("out", 0);
    };
    lr_lbn = state->GetLbn(beta1, beta2, learning_rate_lbn, train_step_lbn,
                           bias_correction_parallel_conf, AddAdamBiasCorrectionLearningRateOp);
  } else {
    lr_lbn = learning_rate_lbn;
  }
  adam_update_op_builder.OpTypeName("adam_update")
      .Input("model", GenLogicalBlobName(var_op->BnInOp2Lbi("out")))
      .Input("model_diff", model_diff_lbn)
      .Input("learning_rate", lr_lbn)
      .Input("m", GenVariableOutputLbn(m_var))
      .Input("v", GenVariableOutputLbn(v_var))
      .Attr<float>("beta1", beta1)
      .Attr<float>("beta2", beta2)
      .Attr<float>("epsilon", epsilon)
      .Attr<float>("weight_decay", GetOptimizerWeightDecayRate(optimizer_conf, *var_op))
      .ScopeSymbolId(var_op->op_conf().scope_symbol_id());
  SetDynamicLossScaleSkipIf(ctx, &adam_update_op_builder);
  const auto adam_update_op = adam_update_op_builder.Build();
  job_builder->AddOps(var_op_node.parallel_desc().parallel_conf(), {adam_update_op.op_conf()});
}

}  // namespace

REGISTER_OPTIMIZER(OptimizerConf::kAdamConf, &GenerateOptimizerOpConf);
REGISTER_OPTIMIZER(OptimizerConf::kLazyAdamConf, &GenerateOptimizerOpConf);

}  // namespace oneflow
