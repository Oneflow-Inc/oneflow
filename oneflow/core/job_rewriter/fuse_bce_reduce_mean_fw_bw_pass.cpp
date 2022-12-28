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

void UpdateConsumerOpConf(const OpNode* consumer, const LogicalBlobId& out,
                          const std::string& new_out_lbn,
                          HashMap<std::string, OperatorConf>* op_name2op_conf) {
  const std::string& consumer_op_name = consumer->op().op_name();
  if (op_name2op_conf->find(consumer_op_name) == op_name2op_conf->end()) {
    (*op_name2op_conf)[consumer_op_name] = consumer->op().op_conf();
  }
  for (const std::string& ibn : consumer->op().input_bns()) {
    if (consumer->op().BnInOp2Lbi(ibn) == out) {
      OperatorConf& consumer_op_conf = op_name2op_conf->at(consumer_op_name);
      const auto& new_val = new_out_lbn;
      const auto& old_val = ReplaceInputLbnInOpCustomizedConf(&consumer_op_conf, ibn, new_val);
      CHECK_EQ(GenLogicalBlobName(out), old_val);
    }
  }
}

class FuseBCEReduceMeanFwBwPass final : public JobPass {
 public:
  FuseBCEReduceMeanFwBwPass() = default;
  ~FuseBCEReduceMeanFwBwPass() override = default;

  bool IsEnabled(const JobPassCtx& ctx) const {
    return ParseBooleanFromEnv("ONEFLOW_FUSE_BCE_REDUCE_MEAN_FW_BW", false);
  }
  Maybe<void> Apply(const OpGraph& op_graph, JobBuilder* job_builder) const;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override {
    if (!IsEnabled(*ctx)) { return Maybe<void>::Ok(); }
    const OpGraph op_graph(*job);
    JobBuilder job_builder(job);
    return Apply(op_graph, &job_builder);
  }
};

Maybe<void> FuseBCEReduceMeanFwBwPass::Apply(const OpGraph& op_graph,
                                             JobBuilder* job_builder) const {
  // This pass fuse binary_cross_entropy_with_logits_reduce_mean and
  // binary_cross_entropy_with_logits_reduce_mean_grad. delete the h2f cast to loss, and the
  // constant_like of dy.
  const auto IsSafeToDelete = MakePredicatorIsSafeToDelete(op_graph);
  HashMap<std::string, OperatorConf> op_name2op_conf;
  std::vector<OperatorConf> delete_ops;
  op_graph.ForEachNode([&](const OpNode* op_node) {
    if (!IsUserOpWithTypeName(op_node->op().op_conf(),
                              "binary_cross_entropy_with_logits_reduce_mean")) {
      return;
    }
    if (op_node->out_edges().size() > 2) { return; }
    bool find_grad_op = false;
    for (const OpEdge* out_edge : op_node->out_edges()) {
      const OpNode* consumer = out_edge->dst_node();
      if (!IsSafeToDelete(consumer)) { return; }
      if (!(IsUserOpWithTypeName(consumer->op().op_conf(), "cast")
            || consumer->op().op_conf().has_constant_like_conf()
            || consumer->op().op_conf().has_output_conf())) {
        return;
      }
      if (consumer->op().op_conf().has_constant_like_conf()) {
        const OpNode* grad_node = consumer->SoleOutEdge()->dst_node();
        if (!IsUserOpWithTypeName(grad_node->op().op_conf(),
                                  "binary_cross_entropy_with_logits_reduce_mean_grad")) {
          return;
        }
        find_grad_op = true;
        if (!IsSafeToDelete(grad_node)) { return; }
      }
    }
    if (!find_grad_op) { return; }
    const user_op::UserOpConfWrapper bce_op_conf(op_node->op().op_conf());
    user_op::UserOpConfWrapperBuilder fused_op_builder(bce_op_conf.op_name());
    fused_op_builder.OpTypeName("fused_bce_reduce_mean_fw_bw")
        .Input("input", bce_op_conf.input("input", 0))
        .Input("target", bce_op_conf.input("target", 0))
        .Output("out")
        .Output("dx");
    for (const OpEdge* out_edge : op_node->out_edges()) {
      const OpNode* consumer = out_edge->dst_node();
      if (IsUserOpWithTypeName(consumer->op().op_conf(), "cast")) {
        const user_op::UserOpConfWrapper cast_conf(consumer->op().op_conf());
        fused_op_builder.Attr<DataType>("out_dtype", cast_conf.attr<DataType>("dtype"));
        // delete cast and update cast consumer's in.
        delete_ops.push_back(consumer->op().op_conf());
        for (const OpEdge* cast_out_edge : consumer->out_edges()) {
          const OpNode* cast_consumer = cast_out_edge->dst_node();
          UpdateConsumerOpConf(cast_consumer, GenLogicalBlobId(cast_conf.output("out", 0)),
                               GenLogicalBlobName(bce_op_conf.op_name(), "out_0"),
                               &op_name2op_conf);
        }
      } else if (consumer->op().op_conf().has_constant_like_conf()) {
        fused_op_builder.Attr<double>(
            "constant_value", consumer->op().op_conf().constant_like_conf().float_operand());
        const OpNode* grad_node = consumer->SoleOutEdge()->dst_node();
        // delete constant_like and grad op, update consumer
        delete_ops.push_back(grad_node->op().op_conf());
        delete_ops.push_back(consumer->op().op_conf());
        const user_op::UserOpConfWrapper grad_conf(grad_node->op().op_conf());
        for (const OpEdge* grad_out_edge : grad_node->out_edges()) {
          const OpNode* grad_consumer = grad_out_edge->dst_node();
          UpdateConsumerOpConf(grad_consumer, GenLogicalBlobId(grad_conf.output("dx", 0)),
                               GenLogicalBlobName(bce_op_conf.op_name(), "dx_0"), &op_name2op_conf);
        }
      } else {
        continue;
      }
    }
    user_op::UserOpConfWrapper fused_op =
        fused_op_builder.ScopeSymbolId(bce_op_conf.op_conf().scope_symbol_id()).Build();
    job_builder->MutOpsOnlyOnce({fused_op.op_conf()});
  });
  job_builder->DelOps(delete_ops);
  for (const auto& pair : op_name2op_conf) { job_builder->MutOpsOnlyOnce({pair.second}); }
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_JOB_PASS("FuseBCEReduceMeanFwBwPass", FuseBCEReduceMeanFwBwPass);

}  // namespace oneflow
