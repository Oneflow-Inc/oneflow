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

class InsertPinnedIdentityOpPass final : public JobPass {
 public:
  InsertPinnedIdentityOpPass() = default;
  ~InsertPinnedIdentityOpPass() override = default;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override;
};

Maybe<std::string> InsertPinnedIdentityOp(JobBuilder* job_builder, const OpGraph& op_graph,
                                          const std::string& lbn) {
  auto lbi = GenLogicalBlobId(lbn);
  const OpNode* node = op_graph.OpNode4OpName(lbi.op_name());
  auto pinned_identity_op =
      user_op::UserOpConfWrapperBuilder(lbi.op_name() + "_" + lbi.blob_name() + "_pinned_identity")
          .Op("pinned_identity")
          .Input("in", lbn)
          .Output("out")
          .ScopeSymbolId(node->op().op_conf().scope_symbol_id())
          .Build();
  const auto& parallel_conf = node->parallel_desc().parallel_conf();
  job_builder->AddOps(parallel_conf, {pinned_identity_op.op_conf()});

  node->ForEachNodeOnOutEdge([&](const OpNode* out_node) {
    for (const std::string& ibn : out_node->op().input_bns()) {
      if (out_node->op().BnInOp2Lbi(ibn) == lbi) {
        if (!CHECK_JUST(job_builder->IsInMutOpTransaction(out_node->op().op_name()))) {
          CHECK_JUST(job_builder->MutOpTransactionMut(out_node->op().op_conf()));
        }
        OperatorConf& mut_consumer_op =
            CHECK_JUST(job_builder->MutOpTransactionGet(out_node->op().op_name()));
        const auto& old_lbn = ReplaceInputLbnInOpCustomizedConf(
            &mut_consumer_op, ibn, pinned_identity_op.output("out", 0));
        CHECK_EQ(old_lbn, GenLogicalBlobName(lbi));
      }
    }
  });
  return pinned_identity_op.output("out", 0);
}

Maybe<void> InsertPinnedIdentityOpPass::Apply(Job* job, JobPassCtx* ctx) const {
  if (!ctx->job_desc().IsTrain()) { return Maybe<void>::Ok(); }
  const OpGraph op_graph(*job);
  JobBuilder job_builder(job);

  HashMap<std::string, std::string> pinned_lbns;
  TrainConf* train_conf = job->mutable_job_conf()->mutable_train_conf();
  // insert after loss
  for (int i = 0; i < train_conf->loss_lbn_size(); ++i) {
    const auto& loss_lbn = train_conf->loss_lbn(i);
    auto it = pinned_lbns.find(loss_lbn);
    if (it == pinned_lbns.end()) {
      const auto& pinned_loss_lbn = JUST(InsertPinnedIdentityOp(&job_builder, op_graph, loss_lbn));
      it = pinned_lbns.emplace(loss_lbn, *pinned_loss_lbn).first;
    }
    train_conf->set_loss_lbn(i, it->second);
  }
  // insert after loss initial gradient
  for (int i = 0; i < train_conf->loss_grad_lbn_size(); ++i) {
    const auto& loss_grad_lbn = train_conf->loss_grad_lbn(i);
    auto it = pinned_lbns.find(loss_grad_lbn);
    if (it == pinned_lbns.end()) {
      const auto& pinned_loss_grad_lbn =
          JUST(InsertPinnedIdentityOp(&job_builder, op_graph, loss_grad_lbn));
      it = pinned_lbns.emplace(loss_grad_lbn, *pinned_loss_grad_lbn).first;
    }
    train_conf->set_loss_grad_lbn(i, it->second);
  }
  // insert after variable gradient
  for (int i = 0; i < train_conf->optimizer_conf_size(); ++i) {
    auto* optimizer_conf = train_conf->mutable_optimizer_conf(i);
    for (int j = 0; j < optimizer_conf->variable_grad_lbns_size(); ++j) {
      const auto& variable_grad_lbn = optimizer_conf->variable_grad_lbns(j);
      if (variable_grad_lbn.empty()) { continue; }
      auto it = pinned_lbns.find(variable_grad_lbn);
      if (it == pinned_lbns.end()) {
        const auto& pinned_variable_grad_lbn =
            JUST(InsertPinnedIdentityOp(&job_builder, op_graph, variable_grad_lbn));
        it = pinned_lbns.emplace(variable_grad_lbn, *pinned_variable_grad_lbn).first;
      }
      optimizer_conf->set_variable_grad_lbns(j, it->second);
    }
  }
  JUST(job_builder.MutOpTransactionCommit());
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_JOB_PASS("InsertPinnedIdentityOpPass", InsertPinnedIdentityOpPass);

}  // namespace oneflow
