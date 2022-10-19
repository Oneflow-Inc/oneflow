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

bool NeedDoPass(const Job* job) {
  return std::any_of(job->net().op().cbegin(), job->net().op().cend(), IsAmpIdentityOp);
}

class PruneAmpWhiteIdentityOpPass final : public JobPass {
 public:
  PruneAmpWhiteIdentityOpPass() = default;
  ~PruneAmpWhiteIdentityOpPass() override = default;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override;
};

Maybe<void> PruneAmpWhiteIdentityOpPass::Apply(Job* job, JobPassCtx* ctx) const {
  if (!ctx->job_desc().prune_amp_white_identity_ops()) { return Maybe<void>::Ok(); }
  if (!NeedDoPass(job)) { return Maybe<void>::Ok(); }
  const OpGraph op_graph(*job);
  JobBuilder job_builder(job);
  HashMap<std::string, OperatorConf> op_name2op_conf;
  HashSet<std::string> del_op_names;

  HashSet<std::string> ctrl_in_op_names;
  op_graph.ForEachNode([&](const OpNode* op_node) {
    for (const std::string& ctrl_in_op_name : op_node->op().op_conf().ctrl_in_op_name()) {
      ctrl_in_op_names.insert(ctrl_in_op_name);
    }
  });

  op_graph.ForEachNode([&](const OpNode* op_node) {
    const std::string op_name = op_node->op().op_name();
    const OperatorConf& op_conf = op_node->op().op_conf();
    if (!IsAmpIdentityOp(op_conf)) { return; }
    if (!op_conf.ctrl_in_op_name().empty()) { return; }
    if (ctrl_in_op_names.find(op_name) != ctrl_in_op_names.end()) { return; }
    if (op_node->in_edges().size() != 1) { return; }
    if (del_op_names.find(op_name) != del_op_names.end()) { return; }
    del_op_names.insert(op_conf.name());

    const OpNode* last_amp_id_op = op_node;
    const OpNode* consumer = last_amp_id_op->SoleOutEdge()->dst_node();
    while (IsAmpIdentityOp(consumer->op().op_conf())) {
      if (del_op_names.insert(consumer->op().op_name()).second) { last_amp_id_op = consumer; }
      consumer = consumer->SoleOutEdge()->dst_node();
    }
    const auto& new_in_lbi = op_node->op().BnInOp2Lbi(op_node->op().SoleIbn());
    const auto& old_in_lbi = last_amp_id_op->op().BnInOp2Lbi(last_amp_id_op->op().SoleObn());

    const auto& consumer_op_name = consumer->op().op_name();
    auto iter = op_name2op_conf.find(consumer_op_name);
    if (iter == op_name2op_conf.end()) {
      iter = op_name2op_conf.emplace(consumer->op().op_name(), consumer->op().op_conf()).first;
    }
    OperatorConf& consumer_op_conf = iter->second;
    for (const std::string& ibn : consumer->op().input_bns()) {
      if (consumer->op().BnInOp2Lbi(ibn) == old_in_lbi) {
        const auto& old_val = ReplaceInputLbnInOpCustomizedConf(&consumer_op_conf, ibn,
                                                                GenLogicalBlobName(new_in_lbi));
        CHECK_EQ(GenLogicalBlobName(old_in_lbi), old_val);
      }
    }
  });

  for (const auto& pair : op_name2op_conf) { job_builder.MutOpsOnlyOnce({pair.second}); }

  job_builder.DelOps(std::vector<std::string>{del_op_names.begin(), del_op_names.end()});

  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_JOB_PASS("PruneAmpWhiteIdentityOpPass", PruneAmpWhiteIdentityOpPass);

}  // namespace oneflow
