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
#include <re2/re2.h>

namespace oneflow {

void GenerateOptimizerOpConfWrapperStruct::Call(JobPassCtx* ctx, const VariableOp& var_op,
                                                const ParallelConf& parallel_conf,
                                                JobBuilder* job_builder,
                                                const LogicalBlobId& diff_lbi_of_var_out) const {
  (*func_)(ctx, var_op, parallel_conf, job_builder, diff_lbi_of_var_out);
}

void GenerateOptimizerOpConfIf(JobPassCtx* ctx, const VariableOp& var_op,
                               const ParallelConf& parallel_conf, JobBuilder* job_builder,
                               const LogicalBlobId& diff_lbi_of_var_out) {
  const auto& train_conf = GlobalJobDesc().job_conf().train_conf();
  auto optimizer_case = train_conf.model_update_conf().normal_mdupdt_case();
  auto* obj = NewObj<int32_t, GenerateOptimizerOpConfWrapperStruct>(optimizer_case);
  obj->Call(ctx, var_op, parallel_conf, job_builder, diff_lbi_of_var_out);
}

void AddOptimizerOpConf(JobPassCtx* ctx, const OpGraph& op_graph, JobBuilder* job_builder,
                        const HashMap<LogicalBlobId, LogicalBlobId>& lbi2diff_lbi) {
  op_graph.ForEachNode([&](OpNode* op_node) {
    const VariableOp* var_op = dynamic_cast<const VariableOp*>(&op_node->op());
    if (var_op == nullptr) { return; }
    if (lbi2diff_lbi.find(var_op->BnInOp2Lbi(var_op->SoleObn())) == lbi2diff_lbi.end()) { return; }

    LogicalBlobId diff_lbi_of_var_out = lbi2diff_lbi.at(var_op->BnInOp2Lbi(var_op->SoleObn()));
    const auto& parallel_desc = op_node->parallel_desc();
    GenerateOptimizerOpConfIf(ctx, *var_op, parallel_desc.parallel_conf(), job_builder,
                              diff_lbi_of_var_out);
  });
}

float GetOptimizerWeightDecayRate(const NormalModelUpdateOpUserConf& model_update_conf,
                                  const VariableOp& op) {
  if (model_update_conf.has_weight_decay_conf()) {
    const WeightDecayConf& weight_decay_conf = model_update_conf.weight_decay_conf();
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

template<typename T>
void ConstructMdUpdtOpConf(const VariableOp& op, const LogicalBlobId& diff_lbi_of_var_out,
                           JobBuilder* job_builder, T* mdupdt_op_conf) {
  const auto& train_conf = job_builder->job().job_conf().train_conf();
  *mdupdt_op_conf->mutable_user_conf() = train_conf.model_update_conf();
  mdupdt_op_conf->set_model_diff(GenLogicalBlobName(diff_lbi_of_var_out));
  mdupdt_op_conf->set_model(GenLogicalBlobName(op.BnInOp2Lbi("out")));
  mdupdt_op_conf->set_train_step(train_conf.train_step_lbn());
  const std::string& primary_lr_lbn = train_conf.primary_lr_lbn();
  const std::string& secondary_lr_lbn = train_conf.secondary_lr_lbn();
  if (op.op_conf().variable_conf().model_name() == "weight") {
    mdupdt_op_conf->set_learning_rate(primary_lr_lbn);
  } else if (op.op_conf().variable_conf().model_name() == "bias") {
    mdupdt_op_conf->set_learning_rate(secondary_lr_lbn);
  } else {
    mdupdt_op_conf->set_learning_rate(primary_lr_lbn);
  }
  const float weight_decay_rate = GetOptimizerWeightDecayRate(train_conf.model_update_conf(), op);
  if (weight_decay_rate != 0) { mdupdt_op_conf->set_weight_decay(weight_decay_rate); }
}

#define INSTANTIATE_CONSTRUCTOR_MDUPDT_OP_CONF(T)                                  \
  template void ConstructMdUpdtOpConf<T>(const VariableOp& op,                     \
                                         const LogicalBlobId& diff_lbi_of_var_out, \
                                         JobBuilder* job_builder, T* mdupdt_op_conf)

}  // namespace oneflow
