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
#include "oneflow/core/job_rewriter/job_pass.h"
#include "oneflow/core/job_rewriter/autograd.h"
#include "oneflow/core/job_rewriter/optimizer.h"
#include "oneflow/core/job_rewriter/calculation_pass.h"
#include "oneflow/core/job/scope.h"
#include "oneflow/core/job/scope.pb.h"
#include "oneflow/core/vm/symbol_storage.h"
#include "oneflow/core/framework/instructions_builder.h"

namespace oneflow {

namespace {

class GenerateOptimizerOpConfs final : public JobPass {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GenerateOptimizerOpConfs);
  GenerateOptimizerOpConfs() = default;
  ~GenerateOptimizerOpConfs() override = default;

  bool IsEnabled(const JobPassCtx& ctx) const { return ctx.job_desc().IsTrain(); }

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override;
};

void FilterCurModelLbi2ModelDiffLbiByName(
    const ::google::protobuf::RepeatedPtrField<std::string>& variables,
    const HashMap<LogicalBlobId, LogicalBlobId>& model_lbi2model_diff_lbi,
    HashMap<LogicalBlobId, LogicalBlobId>* cur_model_lbi2model_diff_lbi) {
  for (const std::string& variable : variables) {
    const LogicalBlobId& lbi = GenLogicalBlobId(variable + "/out");
    if (model_lbi2model_diff_lbi.find(lbi) != model_lbi2model_diff_lbi.end()) {
      (*cur_model_lbi2model_diff_lbi)[lbi] = model_lbi2model_diff_lbi.at(lbi);
    }
  }
}

Maybe<JobBuilder> WithCalculationPassScope(const std::string& pass_name, Job* job,
                                           const std::function<Maybe<void>()>& Handler) {
  HashSet<std::string> exists_op_names;
  for (const auto& op_conf : job->net().op()) {
    CHECK_OR_RETURN(exists_op_names.emplace(op_conf.name()).second);
  }
  JUST(Handler());
  // using a new JobBuilder to avoid bugs caused by MutOnlyOnce
  auto new_job_builder = std::make_shared<JobBuilder>(job);
  HashMap<int64_t, std::vector<const OperatorConf*>> scope_id2op_names;
  const auto& scope_storage = *Singleton<symbol::Storage<Scope>>::Get();
  for (const auto& op_conf : job->net().op()) {
    if (exists_op_names.count(op_conf.name()) > 0) { continue; }
    CHECK_OR_RETURN(op_conf.has_scope_symbol_id());
    OF_RETURN_IF_ERROR(scope_storage.MaybeGet(op_conf.scope_symbol_id())) << op_conf.DebugString();
    scope_id2op_names[op_conf.scope_symbol_id()].emplace_back(&op_conf);
  }
  const auto& GetNewScopeSymbolId = [&](int64_t old_scope_symbol_id) -> Maybe<int64_t> {
    const auto& old_scope = JUST(scope_storage.MaybeGet(old_scope_symbol_id));
    std::shared_ptr<ScopeProto> new_scope = std::make_shared<ScopeProto>(old_scope.scope_proto());
    new_scope->set_parent_scope_symbol_id(old_scope_symbol_id);
    new_scope->set_calculation_pass_name(pass_name);
    std::shared_ptr<Scope> new_scope_symbol;
    JUST(PhysicalRun([&](InstructionsBuilder* builder) -> Maybe<void> {
      new_scope_symbol = JUST(builder->GetScopeSymbol(*new_scope));
      return Maybe<void>::Ok();
    }));
    return JUST(new_scope_symbol->symbol_id());
  };
  for (const auto& pair : scope_id2op_names) {
    int64_t new_scope_symbol_id = JUST(GetNewScopeSymbolId(pair.first));
    std::vector<OperatorConf> op_confs(pair.second.size());
    for (int i = 0; i < pair.second.size(); ++i) {
      op_confs.at(i).CopyFrom(*pair.second.at(i));
      op_confs.at(i).set_scope_symbol_id(new_scope_symbol_id);
    }
    new_job_builder->MutOpsOnlyOnce(op_confs);
  }
  return new_job_builder;
}

Maybe<void> GenerateOptimizerOpConfs::Apply(Job* job, JobPassCtx* ctx) const {
  if (!IsEnabled(*ctx)) { return Maybe<void>::Ok(); }
  const auto& train_conf = job->job_conf().train_conf();
  // loss initial gradients
  HashMap<LogicalBlobId, LogicalBlobId> loss_lbi2initial_diff_lbi;
  CHECK_OR_RETURN(train_conf.loss_lbn_size() == train_conf.loss_grad_lbn_size())
      << "loss_lbn and loss_grad_lbn size mismatch";
  for (int i = 0; i < train_conf.loss_lbn_size(); ++i) {
    auto loss_lbi = GenLogicalBlobId(train_conf.loss_lbn(i));
    auto loss_grad_lbi = GenLogicalBlobId(train_conf.loss_grad_lbn(i));
    loss_lbi2initial_diff_lbi.emplace(loss_lbi, loss_grad_lbi);
  }
  // variable gradients
  HashMap<LogicalBlobId, LogicalBlobId> model_lbi2model_diff_lbi;
  for (const auto& optimizer_conf : train_conf.optimizer_conf()) {
    CHECK_OR_RETURN(optimizer_conf.variable_op_names_size()
                    == optimizer_conf.variable_grad_lbns_size())
        << "variable_op_names and variable_grad_lbns size mismatch";
    for (int i = 0; i < optimizer_conf.variable_op_names_size(); ++i) {
      auto model_lbi = GenLogicalBlobId(optimizer_conf.variable_op_names(i) + "/out");
      const auto& model_diff_lbn = optimizer_conf.variable_grad_lbns(i);
      // variable maybe has no gradient, so skip it if model_diff_lbn is empty
      if (!model_diff_lbn.empty()) {
        model_lbi2model_diff_lbi.emplace(model_lbi, GenLogicalBlobId(model_diff_lbn));
      }
    }
  }
  const OpGraph op_graph(*job);
  auto job_builder = std::make_shared<JobBuilder>(job);
  const JobBuilder* old_job_builder = job_builder.get();
  job_builder = JUST(WithCalculationPassScope(kOptimizerPass, job, [&]() -> Maybe<void> {
    CHECK(old_job_builder == job_builder.get());  // Check this lambda never been async called
    AddDiffHalf2FloatCast(op_graph, job_builder.get(), &model_lbi2model_diff_lbi);
    AddDiffStaticShapeCast(op_graph, job_builder.get(), &model_lbi2model_diff_lbi);
    AddDiffParallelCast(op_graph, job_builder.get(), &model_lbi2model_diff_lbi);
    JUST(ScaleModelDiffByLossInstanceNum(op_graph, job_builder.get(), &model_lbi2model_diff_lbi));
    JUST(ScaleInitialDiffByLossScale(ctx, op_graph, job_builder.get(), &loss_lbi2initial_diff_lbi));
    ScaleModelDiffByLossScale(ctx, op_graph, job_builder.get(), &model_lbi2model_diff_lbi);
    JUST(CountNotFiniteIfNeeded(ctx, op_graph, job_builder.get(), model_lbi2model_diff_lbi));
    for (const auto& optimizer_conf : job->job_conf().train_conf().optimizer_conf()) {
      HashMap<LogicalBlobId, LogicalBlobId> cur_model_lbi2model_diff_lbi;
      FilterCurModelLbi2ModelDiffLbiByName(optimizer_conf.variable_op_names(),
                                           model_lbi2model_diff_lbi, &cur_model_lbi2model_diff_lbi);
      if (optimizer_conf.has_clip_conf()) {
        ClipGradient(ctx, op_graph, job_builder.get(), &cur_model_lbi2model_diff_lbi,
                     optimizer_conf.clip_conf());
      }
      RegularizeGradient(op_graph, job_builder.get(), &cur_model_lbi2model_diff_lbi);
      op_graph.ForEachNode([&](OpNode* op_node) {
        const VariableOp* var_op = dynamic_cast<const VariableOp*>(&op_node->op());
        if (var_op == nullptr
            || cur_model_lbi2model_diff_lbi.find(var_op->BnInOp2Lbi(var_op->SoleObn()))
                   == cur_model_lbi2model_diff_lbi.end()) {
          return;
        }
        const std::string& model_diff_lbn = GenLogicalBlobName(
            cur_model_lbi2model_diff_lbi.at(var_op->BnInOp2Lbi(var_op->SoleObn())));
        AddOptimizerOp(ctx, *op_node, model_diff_lbn, optimizer_conf, job_builder.get());
      });
    }
    return Maybe<void>::Ok();
  }));
  return Maybe<void>::Ok();
}

REGISTER_JOB_PASS("GenerateOptimizerOpConfs", GenerateOptimizerOpConfs);

}  // namespace

}  // namespace oneflow
