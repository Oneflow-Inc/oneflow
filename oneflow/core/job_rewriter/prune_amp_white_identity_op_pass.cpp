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
  using ArgPair = std::pair<const OpNode*, std::string>;  // OpNode and ibn
  HashMap<ArgPair, std::string> to_update_args;
  HashSet<std::string> del_op_names;

  HashSet<std::string> ctrl_in_op_names;
  op_graph.ForEachNode([&](const OpNode* op_node) {
    for (const std::string& ctrl_in_op_name : op_node->op().op_conf().ctrl_in_op_name()) {
      ctrl_in_op_names.insert(ctrl_in_op_name);
    }
  });

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

    // add amp identity to del list
    CHECK(del_op_names.insert(op_name).second);

    std::string in_lbn = GenLogicalBlobName(op_node->op().BnInOp2Lbi(op_node->op().SoleIbn()));
    std::string out_lbn = GenLogicalBlobName(op_node->op().BnInOp2Lbi(op_node->op().SoleObn()));
    // consume a deleted amp identity, find it's preceding lbn
    auto iter = to_update_args.find(ArgPair{op_node, op_node->op().SoleIbn()});
    if (iter != to_update_args.end()) { in_lbn = iter->second; }

    for (const OpEdge* out_edge : op_node->out_edges()) {
      const OpNode* consumer = out_edge->dst_node();
      if (del_op_names.find(consumer->op().op_name()) == del_op_names.end()) {
        for (const std::string& ibn : consumer->op().input_bns()) {
          if (GenLogicalBlobName(consumer->op().BnInOp2Lbi(ibn)) == out_lbn) {
            CHECK(to_update_args.emplace(ArgPair{consumer, ibn}, in_lbn).second);
          }
        }
      } else {
        // comsumer is an already deleted amp identity
        // modify it's succeeding op's input arg lbn
        bool find_arg = false;
        for (auto& arg : to_update_args) {
          if (arg.second == out_lbn) {
            arg.second = in_lbn;
            find_arg = true;
          }
        }
        CHECK(find_arg);
      }
    }
  });

  JobBuilder job_builder(job);
  HashMap<std::string, OperatorConf> op_name2op_conf;
  for (const auto& arg_lbn : to_update_args) {
    const Operator& op = arg_lbn.first.first->op();
    auto iter = op_name2op_conf.find(op.op_name());
    if (iter == op_name2op_conf.end()) {
      iter = op_name2op_conf.emplace(op.op_name(), op.op_conf()).first;
    }
    OperatorConf& op_conf = iter->second;
    const auto& ibn = arg_lbn.first.second;
    auto old_lbn = GenLogicalBlobName(op.BnInOp2Lbi(ibn));
    const auto& old_val = ReplaceInputLbnInOpCustomizedConf(&op_conf, ibn, arg_lbn.second);
    CHECK_EQ(old_lbn, old_val);
  }
  for (const auto& pair : op_name2op_conf) { job_builder.MutOpsOnlyOnce({pair.second}); }
  job_builder.DelOps(std::vector<std::string>{del_op_names.begin(), del_op_names.end()});

  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_JOB_PASS("PruneAmpWhiteIdentityOpPass", PruneAmpWhiteIdentityOpPass);

}  // namespace oneflow
