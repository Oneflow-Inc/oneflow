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
#include "oneflow/core/register/runtime_blob_desc.h"

namespace oneflow {

namespace {

class PruneCastToStaticShapeOpsPass final : public JobPass {
 public:
  PruneCastToStaticShapeOpsPass() = default;
  ~PruneCastToStaticShapeOpsPass() override = default;

  bool IsEnabled(const JobPassCtx& ctx) const {
    return ctx.job_desc().IsTrain() && ctx.job_desc().prune_cast_to_static_shape_ops();
  }
  Maybe<void> Apply(const OpGraph& op_graph, JobBuilder* job_builder) const;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override {
    if (!IsEnabled(*ctx)) { return Maybe<void>::Ok(); }
    const OpGraph op_graph(*job);
    JobBuilder job_builder(job);
    return Apply(op_graph, &job_builder);
  }
};

Maybe<void> PruneCastToStaticShapeOpsPass::Apply(const OpGraph& op_graph,
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
    const OperatorConf& op_conf = op_node->op().op_conf();
    if (!op_conf.has_user_conf()) { return; }
    const std::string& op_type_name = op_conf.user_conf().op_type_name();
    if (op_type_name != "cast_to_static_shape") { return; }
    if (!op_conf.ctrl_in_op_name().empty()) { return; }
    if (ctrl_in_op_names.find(op_conf.name()) != ctrl_in_op_names.end()) { return; }
    if (op_node->in_edges().size() != 1) { return; }
    const user_op::UserOpConfWrapper user_op_conf(op_conf);
    const LogicalBlobId& cast_in_lbi = GenLogicalBlobId(user_op_conf.input("input", 0));
    const LogicalBlobId& cast_out_lbi = GenLogicalBlobId(user_op_conf.output("output", 0));
    const OpNode* producer = op_graph.OpNode4OpName(cast_in_lbi.op_name());
    const BlobDesc& cast_in_logical_blob_desc = producer->LogicalBlobDesc4Lbi(cast_in_lbi);
    if (cast_in_logical_blob_desc.is_dynamic()) { return; }
    for (const OpEdge* out_edge : op_node->out_edges()) {
      const OpNode* consumer = out_edge->dst_node();
      const std::string& consumer_op_name = consumer->op().op_name();
      if (op_name2op_conf.find(consumer_op_name) == op_name2op_conf.end()) {
        op_name2op_conf[consumer_op_name] = consumer->op().op_conf();
      }
      OperatorConf& consumer_op_conf = op_name2op_conf.at(consumer_op_name);
      for (const std::string& ibn : consumer->op().input_bns()) {
        if (consumer->op().BnInOp2Lbi(ibn) == cast_out_lbi) {
          const auto& old_val = ReplaceInputLbnInOpCustomizedConf(&consumer_op_conf, ibn,
                                                                  GenLogicalBlobName(cast_in_lbi));
          CHECK_EQ(GenLogicalBlobName(cast_out_lbi), old_val);
        }
      }
    }
    del_op_names.push_back(op_conf.name());
  });
  for (const auto& pair : op_name2op_conf) { job_builder->MutOpsOnlyOnce({pair.second}); }
  job_builder->DelOps(del_op_names);
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_JOB_PASS("PruneCastToStaticShapeOpsPass", PruneCastToStaticShapeOpsPass);

}  // namespace oneflow
