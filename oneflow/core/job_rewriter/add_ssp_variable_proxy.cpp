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
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/job/scope.h"
#include "oneflow/core/job_rewriter/calculation_pass.h"
#include "oneflow/core/job_rewriter/autograd.h"
#include "oneflow/core/vm/symbol_storage.h"
#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace {

class AddSspVariableProxyPass final : public JobPass {
 public:
  AddSspVariableProxyPass(const AddSspVariableProxyPass&) = delete;
  AddSspVariableProxyPass(AddSspVariableProxyPass&&) = delete;
  AddSspVariableProxyPass() = default;
  ~AddSspVariableProxyPass() = default;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override {
    if (!IsEnabled(*ctx)) { return Maybe<void>::Ok(); }
    const OpGraph op_graph(*job);
    JobBuilder job_builder(job);
    return Apply(op_graph, &job_builder);
  }

  bool IsEnabled(const JobPassCtx& ctx) const {
    return ctx.job_desc().IsTrain() && ctx.job_desc().Bool("enable_ssp");
  }

  Maybe<void> Apply(const OpGraph& op_graph, JobBuilder* job_builder) const {
    HashMap<LogicalBlobId, std::pair<std::string, std::string>> var2ref_value_pair;
    HashSet<OpNode*> var_consumers;
    JUST(ForEachTrainableVarOpNode(op_graph, [&](OpNode* op_node) -> Maybe<void> {
      op_node->ForEachNodeOnOutEdge([&](OpNode* consumer) { var_consumers.insert(consumer); });
      const auto& old_var_out_lbi = op_node->op().BnInOp2Lbi("out");
      return AddSspVarProxyOp(op_node, job_builder, &var2ref_value_pair[old_var_out_lbi].first,
                              &var2ref_value_pair[old_var_out_lbi].second);
    }));
    {
      const auto& NeedReplace = [&](const LogicalBlobId& var_lbi) -> bool {
        return var2ref_value_pair.count(var_lbi) > 0;
      };
      const auto& Ref4Var = [&](const LogicalBlobId& var_lbi) -> const std::string& {
        return var2ref_value_pair.at(var_lbi).first;
      };
      const auto& Val4Var = [&](const LogicalBlobId& var_lbi) -> const std::string& {
        return var2ref_value_pair.at(var_lbi).second;
      };
      for (OpNode* op_node : var_consumers) {
        JUST(ReplaceVarWithSspVarProxyOp(op_node, job_builder, NeedReplace, Ref4Var, Val4Var));
      }
    }
    return Maybe<void>::Ok();
  }

  Maybe<void> ForEachTrainableVarOpNode(const OpGraph& op_graph,
                                        const std::function<Maybe<void>(OpNode*)>& DoEach) const {
    std::function<bool(OpNode*)> NeedBackwardOp;
    JUST(MakePredicatorNeedBackwardOp(op_graph, &NeedBackwardOp));
    const auto& IsSspVarProxy = [](const OperatorConf& op_conf) {
      return op_conf.has_user_conf() && op_conf.user_conf().op_type_name() == "ssp_variable_proxy";
    };
    JUST(op_graph.MaybeForEachNode([&](OpNode* op_node) -> Maybe<void> {
      const auto& op_conf = op_node->op().op_conf();
      CHECK_OR_RETURN(!IsSspVarProxy(op_conf)) << "AddSspVariableProxy can not be applied twice";
      if (op_conf.has_variable_conf() && NeedBackwardOp(op_node)) { return DoEach(op_node); }
      return Maybe<void>::Ok();
    }));
    return Maybe<void>::Ok();
  }

  Maybe<void> AddSspVarProxyOp(OpNode* op_node, JobBuilder* job_builder, std::string* ref_lbn,
                               std::string* value_lbn) const {
    const LogicalBlobId& old_var_out_lbi = op_node->op().BnInOp2Lbi("out");
    int64_t scope_symbol_id = op_node->op().op_conf().scope_symbol_id();
    JUST(AddSspVarProxyOp(old_var_out_lbi, scope_symbol_id, job_builder, ref_lbn, value_lbn));
    return Maybe<void>::Ok();
  }

  Maybe<void> ReplaceVarWithSspVarProxyOp(
      OpNode* op_node, JobBuilder* job_builder,
      const std::function<bool(const LogicalBlobId&)>& NeedReplace,
      const std::function<const std::string&(const LogicalBlobId&)>& Ref4Var,
      const std::function<const std::string&(const LogicalBlobId&)>& Val4Var) const {
    const auto& op = op_node->op();
    std::unique_ptr<std::vector<OperatorConf>> new_op_confs;
    for (const auto& ibn : op.input_bns()) {
      const auto& lbi = op.BnInOp2Lbi(ibn);
      if (!NeedReplace(lbi)) { continue; }
      if (!new_op_confs) { new_op_confs.reset(new std::vector<OperatorConf>({op.op_conf()})); }
      auto* new_op_conf = &new_op_confs->at(0);
      int64_t scope_symbol_id = op.op_conf().scope_symbol_id();
      bool in_optimizer_pass = JUST(IsInOptimizerPass(scope_symbol_id));
      const auto* lbn = (in_optimizer_pass ? &Ref4Var(lbi) : &Val4Var(lbi));
      ReplaceInputLbnInOpCustomizedConf(new_op_conf, ibn, *lbn);
    }
    if (new_op_confs) { job_builder->MutOpsOnlyOnce(*new_op_confs); }
    return Maybe<void>::Ok();
  }

  Maybe<bool> IsInOptimizerPass(int64_t scope_symbol_id) const {
    const auto& scope = JUST(Global<vm::SymbolStorage<Scope>>::Get()->MaybeGet(scope_symbol_id));
    return scope.scope_proto().calculation_pass_name() == kOptimizerPass;
  }

  Maybe<void> AddSspVarProxyOp(const LogicalBlobId& old_var_out_lbi, int64_t scope_symbol_id,
                               JobBuilder* job_builder, std::string* ref_lbn,
                               std::string* value_lbn) const {
    const Scope& scope = JUST(Global<vm::SymbolStorage<Scope>>::Get()->MaybeGet(scope_symbol_id));
    int64_t buffer_size = 0;
    {
      int64_t num_stages = scope.Int64("ssp_num_stages");
      int64_t stage_id = scope.Int64("ssp_stage_id");
      CHECK_GT(num_stages, 0);
      CHECK_GE(stage_id, 0);
      CHECK_LT(stage_id, num_stages);
      buffer_size = num_stages - stage_id;
    }
    std::string op_name = old_var_out_lbi.op_name() + "_ssp_variable_proxy";
    const auto proxy_op = user_op::UserOpConfWrapperBuilder(op_name)
                              .Op("ssp_variable_proxy")
                              .ScopeSymbolId(scope_symbol_id)
                              .Input("var", GenLogicalBlobName(old_var_out_lbi))
                              .Output("ref")
                              .Output("value")
                              .Attr<int64_t>("buffer_size", buffer_size)
                              .Build();
    const auto& parallel_desc = JUST(scope.GetParallelDesc(proxy_op.op_conf()));
    job_builder->AddOps(parallel_desc.parallel_conf(), {proxy_op.op_conf()});
    *ref_lbn = op_name + "/ref_0";
    *value_lbn = op_name + "/value_0";
    return Maybe<void>::Ok();
  }
};

REGISTER_JOB_PASS("AddSspVariableProxy", AddSspVariableProxyPass);

}  // namespace

}  // namespace oneflow
