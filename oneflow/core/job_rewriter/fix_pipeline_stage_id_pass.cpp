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
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/user_op_attr.cfg.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/job_rewriter/job_pass.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/job/scope.h"
#include "oneflow/core/job/scope.cfg.h"
#include "oneflow/core/job_rewriter/calculation_pass.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/vm/vm_util.h"
#include "oneflow/core/vm/symbol_storage.h"

namespace oneflow {

namespace {

class FixPipelineStageIdPass final : public JobPass {
 public:
  OF_DISALLOW_COPY_AND_MOVE(FixPipelineStageIdPass);
  FixPipelineStageIdPass() = default;
  ~FixPipelineStageIdPass() = default;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override {
    if (!IsEnabled(*ctx)) { return Maybe<void>::Ok(); }
    const OpGraph op_graph(*job);
    JobBuilder job_builder(job);
    return Apply(op_graph, &job_builder);
  }

  bool IsEnabled(const JobPassCtx& ctx) const { return ctx.job_desc().IsTrain(); }

  Maybe<void> Apply(const OpGraph& op_graph, JobBuilder* job_builder) const;
};

const Scope& Scope4ScopeSymbolId(int64_t scope_symbol_id) {
  CHECK(Global<symbol::Storage<Scope>>::Get()->Has(scope_symbol_id));
  return Global<symbol::Storage<Scope>>::Get()->Get(scope_symbol_id);
}

const Scope& Scope4OpNode(const OpNode* op_node) {
  const OperatorConf& op_conf = op_node->op().op_conf();
  CHECK(op_conf.has_scope_symbol_id());
  return Scope4ScopeSymbolId(op_conf.scope_symbol_id());
}

bool IsForwardPass(const OpNode* node) {
  return Scope4OpNode(node).scope_proto().calculation_pass_name() == kForwardPass;
}

bool IsBackwardPass(const OpNode* node) {
  return Scope4OpNode(node).scope_proto().calculation_pass_name() == kBackwardPass;
}

bool OpNodeHasScope(const OpNode* node) { return node->op().op_conf().has_scope_symbol_id(); }

int64_t GetStageIdHint(const OpNode* node) {
  return Scope4OpNode(node).Int64("pipeline_stage_id_hint");
}

Maybe<int64_t> NewScopeWithStageId(int64_t old_scope_symbol_id, int64_t stage_id) {
  return NewScopeSymbolId(
      old_scope_symbol_id, [stage_id](std::shared_ptr<cfg::ScopeProto> new_scope) {
        std::cout << "cclog: old scope proto: " << new_scope->DebugString() << "\n\n";
        auto* attr_map = new_scope->mutable_attr_name2attr_value();
        (*attr_map)["pipeline_stage_id_hint"].set_at_int64(stage_id);
        std::cout << "cclog: new scope proto: " << new_scope->DebugString() << "\n\n";
      });
}

Maybe<void> FixPipelineStageIdPass::Apply(const OpGraph& op_graph, JobBuilder* job_builder) const {
  op_graph.ForEachNode([&](const OpNode* this_node) {
    if (!OpNodeHasScope(this_node)) {
      LOG(WARNING) << " op : " << this_node->op().op_conf().DebugString() << " has NOT scope!";
      return;
    }
    int64_t new_scope =
        CHECK_JUST(NewScopeWithStageId(this_node->op().op_conf().scope_symbol_id(), 99));
  });

  if (GlobalJobDesc().job_conf().num_gradient_accumulation_steps() <= 1) {
    return Maybe<void>::Ok();
  }
  int64_t max_stage_id = 0;

  op_graph.ForEachNode([&](const OpNode* this_node) {
    if (!OpNodeHasScope(this_node)) {
      LOG(WARNING) << " op : " << this_node->op().op_conf().DebugString() << " has NOT scope!";
      return;
    }
    max_stage_id = std::max(max_stage_id, GetStageIdHint(this_node));
  });

  if (max_stage_id == 0) { return Maybe<void>::Ok(); }
  const int64_t total_stage_num = max_stage_id + 1;
  LOG(INFO) << "total stage num = " << total_stage_num;

  HashMap<std::string, OperatorConf> mut_op_name2conf;

  for (auto& pair : mut_op_name2conf) { JUST(job_builder->MutOpOnlyOnce(pair.second)); }

  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_JOB_PASS("FixPipelineStageIdPass", FixPipelineStageIdPass);

}  // namespace oneflow
