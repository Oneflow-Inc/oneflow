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
#include "oneflow/core/job_rewriter/dynamic_loss_scale_job_pass_state.h"
//#include "oneflow/core/job_rewriter/adam_bias_correction_learning_rate_state.h"
#include <re2/re2.h>

namespace oneflow {

void GenerateOptimizerOpConfWrapperStruct::Call(JobPassCtx* ctx, const OpNode& var_op_node,
                                                const std::string& model_diff_lbn,
                                                const OptimizerConf& optimizer_conf,
                                                JobBuilder* job_builder) const {
  (*func_)(ctx, var_op_node, model_diff_lbn, optimizer_conf, job_builder);
}

void AddOptimizerOp(JobPassCtx* ctx, const OpNode& var_op_node, const std::string& model_diff_lbn,
                    const OptimizerConf& optimizer_conf, JobBuilder* job_builder) {
  const auto optimizer_case = optimizer_conf.normal_mdupdt_case();
  auto* obj = NewObj<int32_t, GenerateOptimizerOpConfWrapperStruct>(optimizer_case);
  obj->Call(ctx, var_op_node, model_diff_lbn, optimizer_conf, job_builder);
}

void AddEmbeddingUpdateOp(JobPassCtx* ctx, const OpNode& embedding_lookup_op_node,
                          const std::string& model_diff_lbn, const OptimizerConf& optimizer_conf,
                          JobBuilder* job_builder) {
  user_op::UserOpConfWrapper embedding_lookup_op(embedding_lookup_op_node.op().op_conf());
  if (optimizer_conf.has_naive_conf()) {
    auto AddIdentityOp = [&](const std::string& in_lbn, const std::string& op_name,
                             int64_t scope_symbol_id) -> std::string {
      user_op::UserOpConfWrapper identity_op = user_op::UserOpConfWrapperBuilder(op_name)
                                                   .Op("identity")
                                                   .Input("in", in_lbn)
                                                   .Output("out")
                                                   .ScopeSymbolId(scope_symbol_id)
                                                   .Build();
      job_builder->AddOps(embedding_lookup_op_node.parallel_desc().parallel_conf(),
                          {identity_op.op_conf()});
      return identity_op.output("out", 0);
    };
    const std::string& update_op_name = embedding_lookup_op.op_name() + "_update";
    const int64_t scope_symbol_id = embedding_lookup_op.op_conf().scope_symbol_id();
    const std::string& num_unique_indices_lbn =
        AddIdentityOp(embedding_lookup_op.input("num_unique_indices", 0),
                      update_op_name + "_identity_num_unique_indices", scope_symbol_id);
    const std::string& unique_indices_lbn =
        AddIdentityOp(embedding_lookup_op.input("unique_indices", 0),
                      update_op_name + "_identity_unique_indices", scope_symbol_id);
    const std::string& reverse_idx_lbn =
        AddIdentityOp(embedding_lookup_op.input("reverse_idx", 0),
                      update_op_name + "_identity_reverse_idx", scope_symbol_id);

    user_op::UserOpConfWrapperBuilder sgd_update_op_builder(update_op_name);
    sgd_update_op_builder.OpTypeName("sgd_embedding_update")
        .Input("embedding_diff", model_diff_lbn)
        .Input("num_unique_indices", num_unique_indices_lbn)
        .Input("unique_indices", unique_indices_lbn)
        .Input("reverse_idx", reverse_idx_lbn)
        .Input("unique_values", embedding_lookup_op.output("unique_values", 0))
        .Input("learning_rate", optimizer_conf.learning_rate_lbn())
        .Attr<std::string>("name", embedding_lookup_op.attr<std::string>("name"))
        .Attr<int64_t>("embedding_size", embedding_lookup_op.attr<int64_t>("embedding_size"))
        .Attr<float>("weight_decay", 0)  // TODO(guoran):weight_decay
        .ScopeSymbolId(embedding_lookup_op.op_conf().scope_symbol_id());
    SetDynamicLossScaleSkipIf(ctx, &sgd_update_op_builder);
    user_op::UserOpConfWrapper sgd_embedding_update_op = sgd_update_op_builder.Build();
    job_builder->AddOps(embedding_lookup_op_node.parallel_desc().parallel_conf(),
                        {sgd_embedding_update_op.op_conf()});
  } else {
    UNIMPLEMENTED();
  }
}

float GetOptimizerWeightDecayRate(const OptimizerConf& optimizer_conf, const VariableOp& op) {
  if (optimizer_conf.has_weight_decay_conf()) {
    const WeightDecayConf& weight_decay_conf = optimizer_conf.weight_decay_conf();
    std::function<bool(const std::string& op_name)> WeightDecayFilter;
    if (weight_decay_conf.has_includes()) {
      WeightDecayFilter = [&](const std::string& op_name) {
        return std::any_of(
            weight_decay_conf.includes().pattern().cbegin(),
            weight_decay_conf.includes().pattern().cend(),
            [&](const std::string& pattern) { return RE2::PartialMatch(op_name, pattern); });
      };
    } else if (weight_decay_conf.has_excludes()) {
      WeightDecayFilter = [&](const std::string& op_name) {
        return !std::any_of(
            weight_decay_conf.excludes().pattern().cbegin(),
            weight_decay_conf.excludes().pattern().cend(),
            [&](const std::string& pattern) { return RE2::PartialMatch(op_name, pattern); });
      };
    } else {
      WeightDecayFilter = [&](const std::string& op_name) { return true; };
    }
    if (WeightDecayFilter(op.op_name())) {
      return weight_decay_conf.weight_decay_rate();
    } else {
      return 0;
    }
  } else {
    return 0;
  }
}

void SetDynamicLossScaleSkipIf(JobPassCtx* ctx, user_op::UserOpConfWrapperBuilder* builder) {
  if (!ctx->job_desc().job_conf().train_conf().has_dynamic_loss_scale_policy()) { return; }
  builder->Input("skip_if",
                 CHECK_JUST(ctx->GetState<DynamicLossScaleJobPassState>("dynamic_loss_scale_state"))
                     .count_not_finite_lbn());
}

}  // namespace oneflow
