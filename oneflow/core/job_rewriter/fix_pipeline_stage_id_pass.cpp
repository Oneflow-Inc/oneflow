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
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/job_rewriter/job_pass.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/job/scope.h"
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

  bool IsEnabled(const JobPassCtx& ctx) const {
    return ctx.job_desc().IsTrain()
           && ctx.job_desc().job_conf().num_gradient_accumulation_steps() > 1;
  }

  Maybe<void> Apply(const OpGraph& op_graph, JobBuilder* job_builder) const;
};

const Scope& Scope4ScopeSymbolId(int64_t scope_symbol_id) {
  CHECK(Singleton<symbol::Storage<Scope>>::Get()->Has(scope_symbol_id));
  return Singleton<symbol::Storage<Scope>>::Get()->Get(scope_symbol_id);
}

const Scope& Scope4OpNode(const OpNode* op_node) {
  const OperatorConf& op_conf = op_node->op().op_conf();
  CHECK(op_conf.has_scope_symbol_id());
  return Scope4ScopeSymbolId(op_conf.scope_symbol_id());
}

bool OpNodeHasScope(const OpNode* node) { return node->op().op_conf().has_scope_symbol_id(); }

int64_t GetStageIdHint(const OpNode* node) {
  return Scope4OpNode(node).Int64("pipeline_stage_id_hint");
}

std::string ParallelDesc2HashString(const ParallelDesc& parallel_desc) {
  std::string ret = parallel_desc.device_tag() + ",{";
  for (int64_t m : parallel_desc.sorted_machine_ids()) {
    ret += (std::to_string(m) + ":[");
    for (int64_t d : parallel_desc.sorted_dev_phy_ids(m)) { ret += (std::to_string(d) + ","); }
    ret += "],";
  }
  ret += "}";
  return ret;
}

Maybe<int64_t> NewScopeWithStageId(int64_t old_scope_symbol_id, int64_t stage_id) {
  return NewScopeSymbolId(
      old_scope_symbol_id,
      [stage_id](
          std::shared_ptr<ScopeProto> new_scope) {  // NOLINT(performance-unnecessary-value-param)
        auto* attr_map = new_scope->mutable_attr_name2attr_value();
        (*attr_map)["pipeline_stage_id_hint"].set_at_int64(stage_id);
      });
}

Maybe<void> FixPipelineStageIdPass::Apply(const OpGraph& op_graph, JobBuilder* job_builder) const {
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
  VLOG(3) << "total stage num = " << total_stage_num;

  HashMap<std::string, const OpNode*> op_name2node;
  HashMap<std::string, std::vector<const OpNode*>> placement2op_nodes;
  std::vector<OperatorConf> fix_stage_op_confs;

  // NOTE(chengcheng): group op by placement.
  op_graph.ForEachNode([&](const OpNode* this_node) {
    if (!OpNodeHasScope(this_node)) { return; }
    const std::string& op_name = this_node->op().op_name();
    op_name2node.emplace(op_name, this_node);
    std::string placement = ParallelDesc2HashString(this_node->parallel_desc());
    placement2op_nodes[placement].emplace_back(this_node);
  });

  for (auto& pair : placement2op_nodes) {
    int64_t max_stage_id = -1;
    for (const OpNode* this_node : pair.second) {
      max_stage_id = std::max(max_stage_id, GetStageIdHint(this_node));
    }
    CHECK_GE_OR_RETURN(max_stage_id, 0);
    for (const OpNode* this_node : pair.second) {
      int64_t this_stage_id = GetStageIdHint(this_node);
      if (this_stage_id != max_stage_id) {
        VLOG(3) << " In FixPipelineStageIdPass, op_name: " << this_node->op().op_name()
                << " origin_stage_id = " << this_stage_id
                << " is different with same placement : " << pair.first
                << " max_stage_id: " << max_stage_id
                << " , so change this op to the max stage id.\n";
        OperatorConf new_op_conf = this_node->op().op_conf();
        int64_t new_scope_symbol_id =
            JUST(NewScopeWithStageId(new_op_conf.scope_symbol_id(), max_stage_id));
        new_op_conf.set_scope_symbol_id(new_scope_symbol_id);
        fix_stage_op_confs.emplace_back(std::move(new_op_conf));
      }
    }
  }

  for (const auto& op : fix_stage_op_confs) { JUST(job_builder->MutOpOnlyOnce(op)); }
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_JOB_PASS("FixPipelineStageIdPass", FixPipelineStageIdPass);

}  // namespace oneflow
