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
#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace {

class EliminateDeadNodesPass final : public JobPass {
 public:
  EliminateDeadNodesPass() = default;
  ~EliminateDeadNodesPass() override = default;

  Maybe<void> Apply(const OpGraph& op_graph, JobBuilder* job_builder) const;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override {
    const OpGraph op_graph(*job);
    JobBuilder job_builder(job);
    return Apply(op_graph, &job_builder);
  }
};

static bool IsNoSideEffect(const OpNode* op_node) {
  static HashSet<std::string> no_side_effect_ops = {
      "constant", "zeros_like", "ones_like", "repeat", "acc", "pack", "unpack",
  };
  static HashSet<OperatorConf::OpTypeCase> no_side_effect_system_ops = {
      OperatorConf::kDeviceTickConf,
  };
  const auto& op_conf = op_node->op().op_conf();
  if (!op_conf.has_user_conf()) { return no_side_effect_system_ops.count(op_conf.op_type_case()); }
  return no_side_effect_ops.count(op_conf.user_conf().op_type_name());
}

Maybe<void> EliminateDeadNodesPass::Apply(const OpGraph& op_graph, JobBuilder* job_builder) const {
  HashSet<const OpNode*> delete_ops;
  std::vector<OperatorConf> delete_op_confs;
  op_graph.ReverseTopoForEachNode([&](const OpNode* op_node) {
    if (!IsNoSideEffect(op_node)) { return; }
    for (const auto* out_edge : op_node->out_edges()) {
      if (!delete_ops.count(out_edge->dst_node())) { return; }
    }
    VLOG(3) << "Eliminate dead node: " << op_node->op().op_name();
    delete_ops.insert(op_node);
    delete_op_confs.emplace_back(op_node->op().op_conf());
  });

  job_builder->DelOps(delete_op_confs);
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_JOB_PASS("EliminateDeadNodesPass", EliminateDeadNodesPass);

}  // namespace oneflow
