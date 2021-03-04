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
#include "oneflow/core/register/runtime_blob_desc.h"
#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace {

class FuseAddToOutputPass final : public JobPass {
 public:
  FuseAddToOutputPass() = default;
  ~FuseAddToOutputPass() override = default;

  bool IsEnabled(const JobPassCtx& ctx) const {
    return ctx.job_desc().job_conf().enable_fuse_add_to_output();
  }
  Maybe<void> Apply(const OpGraph& op_graph, JobBuilder* job_builder) const;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override {
    if (!IsEnabled(*ctx)) { return Maybe<void>::Ok(); }
    const OpGraph op_graph(*job);
    JobBuilder job_builder(job);
    return Apply(op_graph, &job_builder);
  }
};

Maybe<void> FuseAddToOutputPass::Apply(const OpGraph& op_graph, JobBuilder* job_builder) const {
  const HashMap<std::string, user_op::OpArg> supported_op_type_name2output_arg(
      {{"conv_data_grad", user_op::OpArg("dx", 0)},
       {"normalization", user_op::OpArg("y", 0)},
       {"dropout", user_op::OpArg("out", 0)},
       {"matmul", user_op::OpArg("out", 0)},
       {"layer_norm_grad", user_op::OpArg("dx", 0)},
       {"batch_matmul", user_op::OpArg("out", 0)}});
  HashMap<std::string, OperatorConf> op_name2op_conf;
  auto IsAddToOutputSupported = [&](const OpNode* node, const LogicalBlobId& lbi) -> bool {
    const OperatorConf& op_conf = node->op().op_conf();
    if (!op_conf.has_user_conf()) { return false; }
    if (op_name2op_conf.find(op_conf.name()) != op_name2op_conf.end()) { return false; }
    auto it = supported_op_type_name2output_arg.find(op_conf.user_conf().op_type_name());
    if (it == supported_op_type_name2output_arg.end()) { return false; }
    const user_op::UserOpConfWrapper user_op_conf(op_conf);
    if (GenLogicalBlobId(user_op_conf.output(it->second.name(), it->second.index())) != lbi) {
      return false;
    }
    // add op should be the only consumer
    int64_t output_consumer_cnt = 0;
    for (const OpEdge* out_edge : node->out_edges()) {
      if (std::find(out_edge->lbis().cbegin(), out_edge->lbis().cend(), lbi)
          != out_edge->lbis().cend()) {
        output_consumer_cnt += 1;
      }
    }
    if (output_consumer_cnt != 1) { return false; }
    // already fused
    if (user_op_conf.has_input("_add_to_output", 0)) { return false; }
    return true;
  };
  HashSet<std::string> ctrl_in_op_names;
  op_graph.ForEachNode([&](const OpNode* op_node) {
    for (const std::string& ctrl_in_op_name : op_node->op().op_conf().ctrl_in_op_name()) {
      ctrl_in_op_names.insert(ctrl_in_op_name);
    }
  });

  auto IsReachable = op_graph.MakePredicatorIsOpNameDataOrCtrlReachable();
  op_graph.ForEachNode([&](const OpNode* op_node) {
    const OperatorConf& op_conf = op_node->op().op_conf();
    if (!op_conf.has_user_conf()) { return; }
    if (!op_conf.ctrl_in_op_name().empty()) { return; }
    if (ctrl_in_op_names.find(op_conf.name()) != ctrl_in_op_names.end()) { return; }
    if (op_conf.user_conf().op_type_name() != "add_n") { return; }
    if (op_name2op_conf.find(op_conf.name()) != op_name2op_conf.end()) { return; }
    const user_op::UserOpConfWrapper user_op_conf(op_conf);
    if (user_op_conf.input_size("in") != 2) { return; }

    const LogicalBlobId in_0 = GenLogicalBlobId(user_op_conf.input("in", 0));
    const LogicalBlobId in_1 = GenLogicalBlobId(user_op_conf.input("in", 1));
    const LogicalBlobId out = GenLogicalBlobId(user_op_conf.output("out", 0));
    const OpNode* in_0_node = op_graph.OpNode4OpName(in_0.op_name());
    const OpNode* in_1_node = op_graph.OpNode4OpName(in_1.op_name());

    const OpNode* add_to_node;
    const LogicalBlobId* add_to_lbi;
    const LogicalBlobId* sum_lbi;
    if ((!IsReachable(in_0.op_name(), in_1.op_name())) && IsAddToOutputSupported(in_0_node, in_0)) {
      add_to_node = in_0_node;
      add_to_lbi = &in_1;
      sum_lbi = &in_0;
    } else if ((!IsReachable(in_1.op_name(), in_0.op_name()))
               && IsAddToOutputSupported(in_1_node, in_1)) {
      add_to_node = in_1_node;
      add_to_lbi = &in_0;
      sum_lbi = &in_1;
    } else {
      return;
    }
    OperatorConf new_add_to_op_conf = add_to_node->op().op_conf();
    *(*(new_add_to_op_conf.mutable_user_conf()->mutable_input()))["_add_to_output"]
         .mutable_s()
         ->Add() = GenLogicalBlobName(*add_to_lbi);
    job_builder->MutOpsOnlyOnce({new_add_to_op_conf});
    for (const OpEdge* out_edge : op_node->out_edges()) {
      const OpNode* consumer = out_edge->dst_node();
      const std::string& consumer_op_name = consumer->op().op_name();
      if (op_name2op_conf.find(consumer_op_name) == op_name2op_conf.end()) {
        op_name2op_conf[consumer_op_name] = consumer->op().op_conf();
      }
      for (const std::string& ibn : consumer->op().input_bns()) {
        if (consumer->op().BnInOp2Lbi(ibn) == out) {
          OperatorConf& consumer_op_conf = op_name2op_conf.at(consumer_op_name);
          const auto& new_val = GenLogicalBlobName(*sum_lbi);
          const auto& old_val = ReplaceInputLbnInOpCustomizedConf(&consumer_op_conf, ibn, new_val);
          CHECK_EQ(GenLogicalBlobName(out), old_val);
        }
      }
    }
    job_builder->DelOps({op_conf});
  });
  for (const auto& pair : op_name2op_conf) { job_builder->MutOpsOnlyOnce({pair.second}); }
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_JOB_PASS("FuseAddToOutputPass", FuseAddToOutputPass);

}  // namespace oneflow
