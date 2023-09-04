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

bool IsQunatizationOp(const OperatorConf& op_conf) {
  return op_conf.has_user_conf()
         && (op_conf.user_conf().op_type_name() == "quantization");
}

bool NeedDoPass(const Job& job) {
  return std::any_of(job.net().op().cbegin(), job.net().op().cend(), IsQunatizationOp);
}

class PruneReduntantQuantizationOpsPass final : public JobPass {
 public:
  PruneReduntantQuantizationOpsPass() = default;
  ~PruneReduntantQuantizationOpsPass() override = default;

  bool IsEnabled(const JobPassCtx& ctx) const { return true; }
  Maybe<void> Apply(const OpGraph& op_graph, JobBuilder* job_builder) const;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override {
    if (!IsEnabled(*ctx)) { return Maybe<void>::Ok(); }
    if (!NeedDoPass(*job)) { return Maybe<void>::Ok(); }
    const OpGraph op_graph(*job);
    JobBuilder job_builder(job);
    return Apply(op_graph, &job_builder);
  }
};

Maybe<void> PruneReduntantQuantizationOpsPass::Apply(const OpGraph& op_graph,
                                            JobBuilder* job_builder) const {
  HashMap<std::string, OperatorConf> op_name2op_conf;
  HashSet<std::string> ctrl_in_op_names;
  op_graph.ForEachNode([&](const OpNode* op_node) {
    for (const std::string& ctrl_in_op_name : op_node->op().op_conf().ctrl_in_op_name()) {
      ctrl_in_op_names.insert(ctrl_in_op_name);
    }
  });
  std::vector<std::string> del_op_names;
  op_graph.ForEachNode([&](const OpNode* op_node) {
    if (op_node->out_edges().size() == 1) { return; }
    bool has_found_quant_op = false;
    LogicalBlobId first_quantization_lbi;
    for (const auto* out_edge : op_node->out_edges()) {
      OpNode* consumer = out_edge->dst_node();
      const OperatorConf& op_conf = consumer->op().op_conf();

      if (ctrl_in_op_names.find(op_conf.name()) != ctrl_in_op_names.end()) { return; }
      if (!op_conf.has_user_conf()) { continue; }
      if (op_conf.user_conf().op_type_name() != "quantization") { continue; }
      std::vector<std::string> first_quantization_ctrl_in_op_names;
      user_op::UserOpConfWrapper conf_wrapper(op_conf);
      if (has_found_quant_op) {
        const LogicalBlobId& quantization_lbi = GenLogicalBlobId(conf_wrapper.output("out", 0));
        for (const OpEdge* consumer_out_edge : consumer->out_edges()) {
          const OpNode* consumer = consumer_out_edge->dst_node();
          const std::string& consumer_op_name = consumer->op().op_name();
          if (op_name2op_conf.find(consumer_op_name) == op_name2op_conf.end()) {
            op_name2op_conf[consumer_op_name] = consumer->op().op_conf();
          }
          OperatorConf& consumer_op_conf = op_name2op_conf.at(consumer_op_name);
          for (const std::string& ibn : consumer->op().input_bns()) {
            if (consumer->op().BnInOp2Lbi(ibn) == quantization_lbi) {
              const auto& new_val = GenLogicalBlobName(first_quantization_lbi);
              const auto& old_val = ReplaceInputLbnInOpCustomizedConf(&consumer_op_conf, ibn, new_val);
              CHECK_EQ(GenLogicalBlobName(quantization_lbi), old_val);
              for (const auto& ctrl_in_op_name : op_conf.ctrl_in_op_name()) {
                consumer_op_conf.add_ctrl_in_op_name(ctrl_in_op_name);
              }
            }
          }
          del_op_names.emplace_back(op_conf.name());
        }
      } else {
        first_quantization_lbi = GenLogicalBlobId(conf_wrapper.output("out", 0));
        for (const auto& ctrl_in_op_name : op_conf.ctrl_in_op_name()) {
          first_quantization_ctrl_in_op_names.emplace_back(ctrl_in_op_name);
        }
        has_found_quant_op = true;
      }
    }
  });
  for (const auto& pair : op_name2op_conf) { job_builder->MutOpsOnlyOnce({pair.second}); }
  job_builder->DelOps(del_op_names);
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_JOB_PASS("PruneRedundantQuantizationOpsPass", PruneReduntantQuantizationOpsPass);

}  // namespace oneflow
