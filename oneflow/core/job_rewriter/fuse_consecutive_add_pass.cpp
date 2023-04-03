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
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/job_rewriter/job_pass.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/common/cost_util.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

namespace {

class FuseConsecutiveAddPass final : public JobPass {
 public:
  FuseConsecutiveAddPass() = default;
  ~FuseConsecutiveAddPass() override = default;

  Maybe<void> Apply(const OpGraph& op_graph, JobBuilder* job_builder) const;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override {
    const OpGraph op_graph(*job);
    JobBuilder job_builder(job);
    JUST(Apply(op_graph, &job_builder));
    return Maybe<void>::Ok();
  }
};

Maybe<void> FuseConsecutiveAddPass::Apply(const OpGraph& op_graph, JobBuilder* job_builder) const {
  const auto IsSafeToDelete = MakePredicatorIsSafeToDelete(op_graph);
  std::vector<std::string> delete_ops;
  op_graph.TopoForEachNode([&](const OpNode* op_node) {
    if (!IsUserOpWithTypeName(op_node->op().op_conf(), "add_n") || !IsSafeToDelete(op_node)
        || op_node->out_edges().size() != 1) {
      return;
    }
    OpNode* sole_dst_node = op_node->SoleOutEdge()->dst_node();
    if (!IsUserOpWithTypeName(sole_dst_node->op().op_conf(), "add_n")
        || !IsSafeToDelete(sole_dst_node)) {
      return;
    }

    const std::string this_op_name = op_node->op().op_name();

    const auto& GetCurOpConf = [&](const OpNode& cur_op) -> OperatorConf {
      const std::string& cur_op_name = cur_op.op().op_name();
      if (!CHECK_JUST(job_builder->IsInMutOpTransaction(cur_op_name))) {
        return cur_op.op().op_conf();
      } else {
        return CHECK_JUST(job_builder->MutOpTransactionGet(cur_op_name));
      }
    };

    int64_t fused_cnt = 0;
    auto fused_op_conf = GetCurOpConf(*sole_dst_node);
    auto in_it = fused_op_conf.mutable_user_conf()->mutable_input()->find("in");
    CHECK(in_it != fused_op_conf.mutable_user_conf()->mutable_input()->end());
    auto* in_lbns = in_it->second.mutable_s();
    auto in_lbn_it = in_lbns->begin();
    while (in_lbn_it != in_lbns->end()) {
      const auto lbi = GenLogicalBlobId(*in_lbn_it);
      if (lbi.op_name() == this_op_name) {
        in_lbn_it = in_lbns->erase(in_lbn_it);
        ++fused_cnt;
      } else {
        ++in_lbn_it;
      }
    }

    const auto& this_op_conf = GetCurOpConf(*op_node);
    auto this_in_it = this_op_conf.user_conf().input().find("in");
    CHECK(this_in_it != this_op_conf.user_conf().input().end());
    for (int64_t fuse_i = 0; fuse_i < fused_cnt; ++fuse_i) {
      for (const auto& this_in_lbn : this_in_it->second.s()) { *(in_lbns->Add()) = this_in_lbn; }
    }

    CHECK_JUST(job_builder->MutOpTransactionMut(fused_op_conf));
    delete_ops.emplace_back(this_op_name);
  });

  if (delete_ops.empty()) { return Maybe<void>::Ok(); }
  JUST(job_builder->MutOpTransactionCommit());
  job_builder->DelOps(delete_ops);
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_JOB_PASS("FuseConsecutiveAddPass", FuseConsecutiveAddPass);

}  // namespace oneflow
