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
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/job_rewriter/job_pass.h"

namespace oneflow {

namespace {

bool IsAmpIdentityOp(const OperatorConf& op) {
  return op.has_user_conf()
         && (op.user_conf().op_type_name() == "amp_white_identity"
             || op.user_conf().op_type_name() == "amp_black_identity");
}

bool NeedDoPass(const Job& job) {
  return std::any_of(job.net().op().cbegin(), job.net().op().cend(), IsAmpIdentityOp);
}

class PruneAmpWhiteIdentityOpPass final : public JobPass {
 public:
  PruneAmpWhiteIdentityOpPass() = default;
  ~PruneAmpWhiteIdentityOpPass() override = default;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override;
};

Maybe<void> PruneAmpWhiteIdentityOpPass::Apply(Job* job, JobPassCtx* ctx) const {
  if (!ctx->job_desc().prune_amp_white_identity_ops()) { return Maybe<void>::Ok(); }
  if (!NeedDoPass(*job)) { return Maybe<void>::Ok(); }
  const OpGraph op_graph(*job);

  HashSet<std::string> ctrl_in_op_names;
  op_graph.ForEachNode([&](const OpNode* op_node) {
    for (const std::string& ctrl_in_op_name : op_node->op().op_conf().ctrl_in_op_name()) {
      ctrl_in_op_names.insert(ctrl_in_op_name);
    }
  });

  HashSet<const OpNode*> del_nodes;
  op_graph.ForEachNode([&](const OpNode* op_node) {
    const std::string& op_name = op_node->op().op_name();
    const OperatorConf& op_conf = op_node->op().op_conf();
    // not amp identity op
    if (!IsAmpIdentityOp(op_conf)) { return; }
    // has ctrl in
    if (!op_conf.ctrl_in_op_name().empty()) { return; }
    // is ctrl in of another op
    if (ctrl_in_op_names.find(op_name) != ctrl_in_op_names.end()) { return; }
    // not sole in
    if (op_node->in_edges().size() != 1) { return; }

    del_nodes.insert(op_node);
  });

  HashMap<std::string, OperatorConf> to_update_op_confs;
  std::vector<std::string> del_op_names;
  del_op_names.reserve(del_nodes.size());
  for (const OpNode* op_node : del_nodes) {
    del_op_names.emplace_back(op_node->op().op_name());

    // find first node not deleted
    const OpNode* first = op_node;
    const OpNode* producer = op_node->SoleInEdge()->src_node();
    while (del_nodes.find(producer) != del_nodes.end()) {
      first = producer;
      producer = producer->SoleInEdge()->src_node();
    }

    const auto& old_lbi = op_node->op().BnInOp2Lbi(op_node->op().SoleObn());
    const auto& new_lbi = first->op().BnInOp2Lbi(first->op().SoleIbn());

    for (const OpEdge* out_edge : op_node->out_edges()) {
      const OpNode* consumer = out_edge->dst_node();
      if (del_nodes.find(consumer) == del_nodes.end()) {
        const Operator& op = consumer->op();
        for (const std::string& ibn : op.input_bns()) {
          if (op.BnInOp2Lbi(ibn) == old_lbi) {
            auto iter = to_update_op_confs.find(op.op_name());
            if (iter == to_update_op_confs.end()) {
              iter = to_update_op_confs.emplace(op.op_name(), op.op_conf()).first;
            }
            OperatorConf& op_conf = iter->second;
            const auto& old_val =
                ReplaceInputLbnInOpCustomizedConf(&op_conf, ibn, GenLogicalBlobName(new_lbi));
            CHECK_EQ_OR_RETURN(GenLogicalBlobName(old_lbi), old_val);
          }
        }
      }
    }
  }

  JobBuilder job_builder(job);
  for (const auto& pair : to_update_op_confs) { job_builder.MutOpsOnlyOnce({pair.second}); }
  job_builder.DelOps(del_op_names);

  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_JOB_PASS("PruneAmpWhiteIdentityOpPass", PruneAmpWhiteIdentityOpPass);

}  // namespace oneflow
