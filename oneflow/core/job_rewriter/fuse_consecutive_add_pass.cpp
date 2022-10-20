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

class FuseConsecutiveAddPass final : public JobPass {
 public:
  FuseConsecutiveAddPass() = default;
  ~FuseConsecutiveAddPass() override = default;

  Maybe<bool> Apply(const OpGraph& op_graph, JobBuilder* job_builder) const;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override {
    bool changed = false;
    do {
      const OpGraph op_graph(*job);
      JobBuilder job_builder(job);
      changed = JUST(Apply(op_graph, &job_builder));
    } while (changed);
    return Maybe<void>::Ok();
  }
};

Maybe<bool> FuseConsecutiveAddPass::Apply(const OpGraph& op_graph, JobBuilder* job_builder) const {
  const auto IsSafeToDelete = MakePredicatorIsSafeToDelete(op_graph);
  std::vector<OperatorConf> delete_ops;
  HashSet<const OpNode*> replaced_ops;
  op_graph.ForEachNode([&](const OpNode* op_node) {
    if (!IsUserOpWithTypeName(op_node->op().op_conf(), "add_n") || !IsSafeToDelete(op_node)
        || op_node->out_edges().size() != 1 || replaced_ops.count(op_node)) {
      return;
    }
    OpNode* sole_dst_node = op_node->SoleOutEdge()->dst_node();
    if (!IsUserOpWithTypeName(sole_dst_node->op().op_conf(), "add_n")
        || !IsSafeToDelete(sole_dst_node)) {
      return;
    }

    replaced_ops.insert(sole_dst_node);
    delete_ops.emplace_back(op_node->op().op_conf());

    user_op::UserOpConfWrapperBuilder fused_op_builder(sole_dst_node->op().op_name());
    fused_op_builder.OpTypeName("add_n").Output("out");

    std::vector<LogicalBlobId> operands;
    for (const OpEdge* edge : sole_dst_node->in_edges()) {
      if (edge->src_node() != op_node) {
        operands.insert(operands.end(), edge->lbis().begin(), edge->lbis().end());
      } else {
        for (const OpEdge* src_node_edge : op_node->in_edges()) {
          operands.insert(operands.end(), src_node_edge->lbis().begin(),
                          src_node_edge->lbis().end());
        }
      }
    }
    for (const auto& lbi : operands) { fused_op_builder.Input("in", GenLogicalBlobName(lbi)); }
    OperatorConf new_op_conf = sole_dst_node->op().op_conf();
    *new_op_conf.mutable_user_conf() = fused_op_builder.Build().op_conf().user_conf();
    job_builder->MutOpsOnlyOnce({new_op_conf});
  });

  if (delete_ops.empty()) { return /*changed = */ false; }
  job_builder->DelOps(delete_ops);
  return /*changed = */ true;
}

}  // namespace

REGISTER_JOB_PASS("FuseConsecutiveAddPass", FuseConsecutiveAddPass);

}  // namespace oneflow
