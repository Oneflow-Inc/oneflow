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
#include "oneflow/core/job/scope.cfg.h"
#include "oneflow/core/job/scope.pb.h"
#include "oneflow/core/job/foreign_callback.h"
#include "oneflow/core/vm/symbol_storage.h"
#include "oneflow/core/framework/interpreter.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/framework/symbol_id_cache.h"

namespace oneflow {

namespace {

void UpdateJobHelperConfProducedLbi2ConsumedDiffLbi(
    const HashMap<LogicalBlobId, LogicalBlobId>& lbi2diff_lbi, JobBuilder* job_builder) {
  auto& mut_pairs =
      (*job_builder->mutable_helper()->mutable_tag2lbi_relations())[kProducedLbi2ConsumedDiffLbi];
  for (const auto& pair : lbi2diff_lbi) {
    auto* mut_pair = mut_pairs.add_pair();
    *mut_pair->mutable_first() = pair.first;
    *mut_pair->mutable_second() = pair.second;
  }
}

void BindIdenticalSbpObaPairsBetweenIbns(const OpNode& op_node, JobBuilder* job_builder) {
  HashMap<LogicalBlobId, std::vector<OpBlobArg>> in_lbi2obas;
  for (const std::string& ibn : op_node.op().input_bns()) {
    in_lbi2obas[op_node.op().BnInOp2Lbi(ibn)].push_back(GenOpBlobArg(op_node.op().op_name(), ibn));
  }
  for (const auto& pair : in_lbi2obas) {
    if (pair.second.size() > 1) {
      FOR_RANGE(int32_t, i, 1, pair.second.size()) {
        job_builder->BindIdenticalSbpOpBlobArgPair(pair.second.at(0), pair.second.at(i));
      }
    }
  }
}

void SetSbpSignatureHintByIdenticalSbpObaPairs(const OpGraph& op_graph, JobBuilder* job_builder) {
  HashMap<OpBlobArg, const SbpParallel*> oba2sbp_parallel;
  op_graph.ForEachNode([&](OpNode* op_node) {
    auto ForEachBn = [&](const std::function<void(const std::string&)>& Handler) {
      for (const auto& ibn : op_node->op().input_bns()) { Handler(ibn); }
      for (const auto& obn : op_node->op().output_bns()) { Handler(obn); }
    };
    ForEachBn([&](const std::string& bn_in_op) {
      const auto& oba = GenOpBlobArg(op_node->op().op_name(), bn_in_op);
      oba2sbp_parallel[oba] = &op_node->SbpParallel4Lbi(op_node->op().BnInOp2Lbi(bn_in_op));
    });
  });
  auto HasSbpParallel = [&](const OpBlobArg& oba) {
    return oba2sbp_parallel.find(oba) != oba2sbp_parallel.end();
  };
  for (const auto& pair : job_builder->job().helper().identical_sbp_oba_pairs().pair()) {
    const SbpParallel* sbp_parallel = nullptr;
    if (HasSbpParallel(pair.first()) && HasSbpParallel(pair.second())) {
      CHECK(oba2sbp_parallel.at(pair.first()) == oba2sbp_parallel.at(pair.second()));
      sbp_parallel = oba2sbp_parallel.at(pair.first());
    } else if (HasSbpParallel(pair.first())) {
      sbp_parallel = oba2sbp_parallel.at(pair.first());
    } else if (HasSbpParallel(pair.second())) {
      sbp_parallel = oba2sbp_parallel.at(pair.second());
    } else {
      UNIMPLEMENTED();
    }
    *job_builder->MutSbpParallel4Oba(pair.first()) = *sbp_parallel;
    *job_builder->MutSbpParallel4Oba(pair.second()) = *sbp_parallel;
  }
}

void UpdateOpSbpSignatureHint(const OpGraph& op_graph, JobBuilder* job_builder) {
  op_graph.ForEachNode(
      [&](OpNode* op_node) { BindIdenticalSbpObaPairsBetweenIbns(*op_node, job_builder); });
  SetSbpSignatureHintByIdenticalSbpObaPairs(op_graph, job_builder);
}

class GenerateBackwardAndOptimizerOpConfs final : public JobPass {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GenerateBackwardAndOptimizerOpConfs);
  GenerateBackwardAndOptimizerOpConfs() = default;
  ~GenerateBackwardAndOptimizerOpConfs() override = default;

  bool IsEnabled(const JobPassCtx& ctx) const { return ctx.job_desc().IsTrain(); }

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override;
};

void FilterModelLbi2ModelDiffLbiByOpConf(
    const OpGraph& op_graph, const HashMap<LogicalBlobId, LogicalBlobId>& lbi2diff_lbi,
    HashMap<LogicalBlobId, LogicalBlobId>* model_lbi2model_diff_lbi) {
  for (const auto& pair : lbi2diff_lbi) {
    const LogicalBlobId& lbi = pair.first;
    const LogicalBlobId& diff_lbi = pair.second;
    const OpNode* producer = op_graph.OpNode4OpName(lbi.op_name());
    if (producer->op().op_conf().has_variable_conf()) {
      (*model_lbi2model_diff_lbi)[lbi] = diff_lbi;
    }
  }
}

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

// TODO(lixinqi): Refactor this function after symbol::IdCache and symbol::Storage merged
template<typename SymbolConfT, typename SymbolPbT, typename SymbolT>
Maybe<void> TryAddSymbol(int64_t symbol_id, const SymbolConfT& symbol_conf) {
  SymbolPbT symbol_pb;
  symbol_conf.ToProto(&symbol_pb);
  auto* id_cache = Global<symbol::IdCache<SymbolConfT>>::Get();
  if (id_cache->Has(symbol_conf)) { return Maybe<void>::Ok(); }
  JUST(id_cache->FindOrCreate(symbol_conf, [&symbol_id]() -> Maybe<int64_t> { return symbol_id; }));
  JUST(Global<symbol::Storage<SymbolT>>::Get()->TryAdd(symbol_id, symbol_pb));
  return Maybe<void>::Ok();
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
  const auto& scope_storage = *Global<symbol::Storage<Scope>>::Get();
  for (const auto& op_conf : job->net().op()) {
    if (exists_op_names.count(op_conf.name()) > 0) { continue; }
    CHECK_OR_RETURN(op_conf.has_scope_symbol_id());
    OF_RETURN_IF_ERROR(scope_storage.MaybeGet(op_conf.scope_symbol_id())) << op_conf.DebugString();
    scope_id2op_names[op_conf.scope_symbol_id()].push_back(&op_conf);
  }
  const auto& GetNewScopeSymbolId = [&](int64_t old_scope_symbol_id) -> Maybe<int64_t> {
    const auto& old_scope = JUST(scope_storage.MaybeGet(old_scope_symbol_id));
    std::shared_ptr<cfg::ScopeProto> new_scope = std::make_shared<cfg::ScopeProto>();
    new_scope->InitFromProto(old_scope.scope_proto());
    new_scope->set_parent_scope_symbol_id(old_scope_symbol_id);
    new_scope->set_calculation_pass_name(pass_name);
    int64_t symbol_id = 0;
    JUST(LogicalInterpreter().Run([&](InstructionsBuilder* builder) -> Maybe<void> {
      symbol_id = JUST(builder->FindOrCreateSymbolId<cfg::ScopeProto>(*new_scope));
      return Maybe<void>::Ok();
    }));
    JUST(TryAddSymbol<cfg::ScopeProto, ScopeProto, Scope>(symbol_id, *new_scope));
    return symbol_id;
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

Maybe<void> GenerateBackwardAndOptimizerOpConfs::Apply(Job* job, JobPassCtx* ctx) const {
  if (!IsEnabled(*ctx)) { return Maybe<void>::Ok(); }
  const OpGraph op_graph(*job);
  auto job_builder = std::make_shared<JobBuilder>(job);
  const JobBuilder* old_job_builder = job_builder.get();
  LogicalBlobId total_loss_instance_num;
  HashMap<LogicalBlobId, LogicalBlobId> lbi2diff_lbi;
  job_builder = JUST(WithCalculationPassScope(kBackwardPass, job, [&]() -> Maybe<void> {
    CHECK(old_job_builder == job_builder.get());  // Check this lambda never been async called
    JUST(AutoGrad(ctx, op_graph, job_builder.get(), &lbi2diff_lbi));
    return Maybe<void>::Ok();
  }));
  HashMap<LogicalBlobId, LogicalBlobId> model_lbi2model_diff_lbi;
  FilterModelLbi2ModelDiffLbiByOpConf(op_graph, lbi2diff_lbi, &model_lbi2model_diff_lbi);
  old_job_builder = job_builder.get();
  job_builder = JUST(WithCalculationPassScope(kOptimizerPass, job, [&]() -> Maybe<void> {
    CHECK(old_job_builder == job_builder.get());  // Check this lambda never been async called
    AddDiffStaticShapeCast(op_graph, job_builder.get(), &model_lbi2model_diff_lbi);
    AddDiffParallelCast(op_graph, job_builder.get(), &model_lbi2model_diff_lbi);
    JUST(ScaleModelDiffByLossInstanceNum(op_graph, job_builder.get(), &model_lbi2model_diff_lbi));
    ScaleModelDiffByLossScale(ctx, op_graph, job_builder.get(), &model_lbi2model_diff_lbi);
    JUST(CountNotFiniteIfNeeded(ctx, op_graph, job_builder.get(), model_lbi2model_diff_lbi));
    RegularizeGradient(op_graph, job_builder.get(), &model_lbi2model_diff_lbi);
    for (const auto& optimizer_conf : job->job_conf().train_conf().optimizer_conf()) {
      HashMap<LogicalBlobId, LogicalBlobId> cur_model_lbi2model_diff_lbi;
      FilterCurModelLbi2ModelDiffLbiByName(optimizer_conf.variable_op_names(),
                                           model_lbi2model_diff_lbi, &cur_model_lbi2model_diff_lbi);
      if (optimizer_conf.has_clip_conf()) {
        ClipGradient(op_graph, job_builder.get(), &cur_model_lbi2model_diff_lbi,
                     optimizer_conf.clip_conf());
      }
      op_graph.ForEachNode([&](OpNode* op_node) {
        const VariableOp* var_op = dynamic_cast<const VariableOp*>(&op_node->op());
        if (var_op == nullptr
            || cur_model_lbi2model_diff_lbi.find(var_op->BnInOp2Lbi(var_op->SoleObn()))
                   == lbi2diff_lbi.end()) {
          return;
        }
        const std::string& model_diff_lbn = GenLogicalBlobName(
            cur_model_lbi2model_diff_lbi.at(var_op->BnInOp2Lbi(var_op->SoleObn())));
        AddOptimizerOp(ctx, *op_node, model_diff_lbn, optimizer_conf, job_builder.get());
      });
    }
    return Maybe<void>::Ok();
  }));
  UpdateJobHelperConfProducedLbi2ConsumedDiffLbi(lbi2diff_lbi, job_builder.get());
  UpdateOpSbpSignatureHint(op_graph, job_builder.get());
  return Maybe<void>::Ok();
}

REGISTER_JOB_PASS("GenerateBackwardAndOptimizerOpConfs", GenerateBackwardAndOptimizerOpConfs);

}  // namespace

}  // namespace oneflow
