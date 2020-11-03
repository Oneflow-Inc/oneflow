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
    JUST(ForEachTrainableVarOpNode(op_graph, [&](OpNode* op_node) -> Maybe<void> {
      return ReplaceVarWithSspVarProxyOp(op_node, job_builder);
    }));
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

  Maybe<void> ReplaceVarWithSspVarProxyOp(OpNode* op_node, JobBuilder* job_builder) const {
    const LogicalBlobId& old_var_out_lbi = op_node->op().BnInOp2Lbi("out");
    std::string ref_lbn;
    std::string value_lbn;
    const Scope* scope = nullptr;
    {
      int64_t scope_symbol_id = op_node->op().op_conf().scope_symbol_id();
      scope = &JUST(Global<vm::SymbolStorage<Scope>>::Get()->MaybeGet(scope_symbol_id));
    }
    JUST(AddSspVarProxyOp(old_var_out_lbi, *scope, job_builder, &ref_lbn, &value_lbn));
    JUST(op_node->ForEachOutNode([&](OpNode* consumer) -> Maybe<void> {
      const auto& op = consumer->op();
      std::vector<OperatorConf> new_op_confs({op.op_conf()});
      auto* new_op_conf = &new_op_confs.at(0);
      for (const auto& ibn : op.input_bns()) {
        if (op.BnInOp2Lbi(ibn) != old_var_out_lbi) { continue; }
        // TODO(lixinqi): variable should be always consumed mutablely by optimizer
        const auto* lbn = (op.InputBlobModifier4Ibn(ibn).is_mutable() ? &ref_lbn : &value_lbn);
        ReplaceInputLbnInOpCustomizedConf(new_op_conf, ibn, *lbn);
      }
      job_builder->MutOpsOnlyOnce(new_op_confs);
      return Maybe<void>::Ok();
    }));
    return Maybe<void>::Ok();
  }

  Maybe<void> AddSspVarProxyOp(const LogicalBlobId& old_var_out_lbi, const Scope& scope,
                               JobBuilder* job_builder, std::string* ref_lbn,
                               std::string* value_lbn) const {
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
                              .ScopeSymbolId(scope.scope_proto().symbol_id())
                              .Input("var", GenLogicalBlobName(old_var_out_lbi))
                              .Output("ref")
                              .Output("value")
                              .Attr<int64_t>("buffer_size", buffer_size)
                              .Build();
    const auto& parallel_desc = JUST(scope.GetParallelDesc(proxy_op.op_conf()));
    job_builder->AddOps(parallel_desc.parallel_conf(), {proxy_op.op_conf()});
    return Maybe<void>::Ok();
  }
};

REGISTER_JOB_PASS("AddSspVariableProxy", AddSspVariableProxyPass);

}  // namespace

}  // namespace oneflow
