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

class PrunePinnedIdentityOpPass final : public JobPass {
 public:
  PrunePinnedIdentityOpPass() = default;
  ~PrunePinnedIdentityOpPass() override = default;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override;
};

Maybe<std::string> PrunePinnedIdentityOp(JobBuilder* job_builder,
                                         std::vector<std::string>* outdated_ops,
                                         const OpGraph& op_graph, const std::string& lbn) {
  auto lbi = GenLogicalBlobId(lbn);
  const OpNode* op_node = op_graph.OpNode4OpName(lbi.op_name());
  CHECK_EQ_OR_RETURN(op_node->in_edges().size(), 1);  // NOLINT
  const OperatorConf& op_conf = op_node->op().op_conf();
  CHECK_OR_RETURN(op_conf.has_user_conf());  // NOLINT
  const std::string& op_type_name = op_conf.user_conf().op_type_name();
  CHECK_OR_RETURN(op_type_name == "pinned_identity");  // NOLINT

  // skip prune if the pinned identity has `ctrl_in_op`
  if (!op_conf.ctrl_in_op_name().empty()) { return lbn; }

  const user_op::UserOpConfWrapper user_op_conf(op_conf);
  const LogicalBlobId& in_lbi = GenLogicalBlobId(user_op_conf.input("in", 0));
  const LogicalBlobId& out_lbi = GenLogicalBlobId(user_op_conf.output("out", 0));

  op_node->ForEachNodeOnOutEdge([&](const OpNode* out_node) {
    for (const std::string& ibn : out_node->op().input_bns()) {
      if (out_node->op().BnInOp2Lbi(ibn) == out_lbi) {
        if (!CHECK_JUST(job_builder->IsInMutOpTransaction(out_node->op().op_name()))) {
          CHECK_JUST(job_builder->MutOpTransactionMut(out_node->op().op_conf()));
        }
        OperatorConf& mut_consumer_op =
            CHECK_JUST(job_builder->MutOpTransactionGet(out_node->op().op_name()));
        const auto& old_lbn =
            ReplaceInputLbnInOpCustomizedConf(&mut_consumer_op, ibn, GenLogicalBlobName(in_lbi));
        CHECK_EQ(old_lbn, GenLogicalBlobName(out_lbi));
      }
    }
  });
  outdated_ops->push_back(op_conf.name());
  return GenLogicalBlobName(in_lbi);
}

Maybe<void> PrunePinnedIdentityOpPass::Apply(Job* job, JobPassCtx* ctx) const {
  if (!job->job_conf().has_train_conf()) { return Maybe<void>::Ok(); }
  const OpGraph op_graph(*job);
  JobBuilder job_builder(job);
  HashMap<std::string, std::string> pruned_lbns;
  std::vector<std::string> outdated_ops;
  TrainConf* train_conf = job->mutable_job_conf()->mutable_train_conf();
  // prune loss pinned identity
  for (int i = 0; i < train_conf->loss_lbn_size(); ++i) {
    const auto& pinned_loss_lbn = train_conf->loss_lbn(i);
    auto it = pruned_lbns.find(pinned_loss_lbn);
    if (it == pruned_lbns.end()) {
      const auto& loss_lbn =
          JUST(PrunePinnedIdentityOp(&job_builder, &outdated_ops, op_graph, pinned_loss_lbn));
      it = pruned_lbns.emplace(pinned_loss_lbn, *loss_lbn).first;
    }
    train_conf->set_loss_lbn(i, it->second);
  }
  // prune loss initial gradient pinned identity
  for (int i = 0; i < train_conf->loss_grad_lbn_size(); ++i) {
    const auto& pinned_loss_grad_lbn = train_conf->loss_grad_lbn(i);
    auto it = pruned_lbns.find(pinned_loss_grad_lbn);
    if (it == pruned_lbns.end()) {
      const auto& loss_grad_lbn =
          JUST(PrunePinnedIdentityOp(&job_builder, &outdated_ops, op_graph, pinned_loss_grad_lbn));
      it = pruned_lbns.emplace(pinned_loss_grad_lbn, *loss_grad_lbn).first;
    }
    train_conf->set_loss_grad_lbn(i, it->second);
  }
  // prune variable gradient pinned identity
  for (int i = 0; i < train_conf->optimizer_conf_size(); ++i) {
    auto* optimizer_conf = train_conf->mutable_optimizer_conf(i);
    for (int j = 0; j < optimizer_conf->variable_grad_lbns_size(); ++j) {
      const auto& pinned_variable_grad_lbn = optimizer_conf->variable_grad_lbns(j);
      if (pinned_variable_grad_lbn.empty()) { continue; }
      auto it = pruned_lbns.find(pinned_variable_grad_lbn);
      if (it == pruned_lbns.end()) {
        const auto& variable_grad_lbn = JUST(
            PrunePinnedIdentityOp(&job_builder, &outdated_ops, op_graph, pinned_variable_grad_lbn));
        it = pruned_lbns.emplace(pinned_variable_grad_lbn, *variable_grad_lbn).first;
      }
      optimizer_conf->set_variable_grad_lbns(j, it->second);
    }
  }
  job_builder.DelOps(outdated_ops);
  JUST(job_builder.MutOpTransactionCommit());
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_JOB_PASS("PrunePinnedIdentityOpPass", PrunePinnedIdentityOpPass);

}  // namespace oneflow
