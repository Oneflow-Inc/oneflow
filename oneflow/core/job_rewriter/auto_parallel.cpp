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
#include <chrono>
#include "oneflow/core/common/hash_container.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/job_rewriter/job_pass.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/auto_parallel/sbp_constructor.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"

namespace oneflow {

namespace {

class AutoParallelPass final : public JobPass {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AutoParallelPass);
  AutoParallelPass() = default;
  ~AutoParallelPass() override = default;

  Maybe<void> Apply(const OpGraph& op_graph, Job* job) const;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override {
    if (!job->job_conf().enable_auto_parallel()) { return Maybe<void>::Ok(); }
    VLOG(3) << "=== Enable AutoParallel ===";
    if (job->job_conf().enable_auto_parallel_ignore_user_sbp_config()) {
      JUST(RemoveParallelCastOps(job));
    }
    const OpGraph op_graph(*job);
    return Apply(op_graph, job);
  }

 private:
  Maybe<void> RemoveParallelCastOps(Job* job) const;
};

Maybe<void> AutoParallelPass::Apply(const OpGraph& op_graph, Job* job) const {
  // auto-parallel
  LOG(INFO) << "Start Auto Parallel";
  auto time_begin = std::chrono::high_resolution_clock::now();

  auto_parallel::SbpConstructor sbp_constructor(op_graph, job);
  JUST(sbp_constructor.FindBestSbpSignature());
  JUST(sbp_constructor.DumpNdSbpSignatureForJob(op_graph, job));
  auto time_end = std::chrono::high_resolution_clock::now();
  VLOG(2) << "Auto parallel took "
          << std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_begin).count()
          << " ms\n";
  if (GlobalProcessCtx::Rank() == 0) {
    // sbp_constructor.PrintSBPGraphDebugInfo();
    JUST(sbp_constructor.CheckSbpAgreement(*job));
  }
  return Maybe<void>::Ok();
}

REGISTER_JOB_PASS("AutoParallelPass", AutoParallelPass);

Maybe<void> AutoParallelPass::RemoveParallelCastOps(Job* job) const {
  VLOG(3) << "Remove parallel cast ops for auto_parallel:";
  const OpGraph op_graph(*job);
  JobBuilder job_builder(job);
  HashMap<std::string, OperatorConf> op_name2op_conf;
  HashMap<std::string, NdSbpSignature> op_name2nd_sbp_signature;
  HashSet<std::string> ctrl_in_op_names;
  op_graph.ForEachNode([&](const OpNode* op_node) {
    for (const std::string& ctrl_in_op_name : op_node->op().op_conf().ctrl_in_op_name()) {
      ctrl_in_op_names.insert(ctrl_in_op_name);
    }
  });
  const auto IsParallelCastOp = [](const OperatorConf& op_conf) -> bool {
    return op_conf.has_user_conf()
           && (op_conf.user_conf().op_type_name() == "parallel_cast"
               || op_conf.user_conf().op_type_name() == "hierarchical_parallel_cast"
               || op_conf.user_conf().op_type_name() == "hierarchical_parallel_cast_like");
  };
  std::vector<std::string> del_op_names;
  HashSet<std::string> del_op_name_set;
  std::function<void(const OpNode*)> Try2Delete = [&](const OpNode* op_node) {
    if (del_op_name_set.find(op_node->op().op_name()) != del_op_name_set.end()) { return; }
    const OperatorConf& op_conf = op_node->op().op_conf();
    if (!IsParallelCastOp(op_conf)) { return; }
    if (!op_conf.ctrl_in_op_name().empty()) {
      VLOG(3) << "Skip " << op_conf.name() << ", because it has ctrl edge.";
      return;
    }
    if (ctrl_in_op_names.find(op_conf.name()) != ctrl_in_op_names.end()) {
      VLOG(3) << "Skip " << op_conf.name() << ", because it is a ctrl edge.";
      return;
    }
    if (op_node->in_edges().size() != 1) { return; }

    // Find the first op which won't be deleted
    const OpNode* source_op = op_node;
    const OpNode* producer = op_node->SoleInEdge()->src_node();
    while (IsParallelCastOp(producer->op().op_conf())) {
      Try2Delete(producer);
      if (del_op_name_set.find(producer->op().op_name()) == del_op_name_set.end()) { break; }
      source_op = producer;
      producer = source_op->SoleInEdge()->src_node();
    }
    user_op::UserOpConfWrapper conf_wrapper_in(source_op->op().op_conf());
    const LogicalBlobId& parallel_cast_in_lbi = GenLogicalBlobId(conf_wrapper_in.input("in", 0));

    user_op::UserOpConfWrapper conf_wrapper_out(op_conf);
    const LogicalBlobId& parallel_cast_out_lbi =
        GenLogicalBlobId(conf_wrapper_out.output("out", 0));
    if (op_node->parallel_desc() != producer->parallel_desc()) {
      VLOG(3) << "Skip " << op_node->op().op_name() << "(with placement: "
              << *CHECK_JUST(PlacementToString(SymbolOf(op_node->parallel_desc())))
              << "), because producer " << producer->op().op_name() << "'s placement is "
              << *CHECK_JUST(PlacementToString(SymbolOf(producer->parallel_desc())));
      return;
    }
    for (const OpEdge* out_edge : op_node->out_edges()) {
      const OpNode* consumer = out_edge->dst_node();
      if (consumer->parallel_desc() != op_node->parallel_desc()) {
        VLOG(3) << "Skip " << op_node->op().op_name() << "(with placement: "
                << *CHECK_JUST(PlacementToString(SymbolOf(op_node->parallel_desc())))
                << "), because consumer " << consumer->op().op_name() << "'s placement is "
                << *CHECK_JUST(PlacementToString(SymbolOf(consumer->parallel_desc())));
        return;
      }
    }
    op_name2nd_sbp_signature[producer->op().op_name()] = producer->nd_sbp_signature();
    for (const OpEdge* out_edge : op_node->out_edges()) {
      const OpNode* consumer = out_edge->dst_node();
      const std::string& consumer_op_name = consumer->op().op_name();
      op_name2nd_sbp_signature[consumer_op_name] = consumer->nd_sbp_signature();
      if (op_name2op_conf.find(consumer_op_name) == op_name2op_conf.end()) {
        op_name2op_conf[consumer_op_name] = consumer->op().op_conf();
      }
      OperatorConf& consumer_op_conf = op_name2op_conf.at(consumer_op_name);
      for (const std::string& ibn : consumer->op().input_bns()) {
        if (consumer->op().BnInOp2Lbi(ibn) == parallel_cast_out_lbi) {
          const auto& new_val = GenLogicalBlobName(parallel_cast_in_lbi);
          const auto& old_val = ReplaceInputLbnInOpCustomizedConf(&consumer_op_conf, ibn, new_val);
          CHECK_EQ(GenLogicalBlobName(parallel_cast_out_lbi), old_val);
        }
      }
    }
    del_op_names.emplace_back(op_conf.name());
    del_op_name_set.insert(op_conf.name());
    VLOG(3) << "\tremove " << op_conf.name();
  };
  op_graph.ForEachNode(Try2Delete);
  for (const auto& pair : op_name2op_conf) { job_builder.MutOpsOnlyOnce({pair.second}); }
  for (const auto& pair : op_name2nd_sbp_signature) {
    job_builder.AddNdSbpSignature4OpName(pair.first, pair.second);
  }
  job_builder.DelOps(del_op_names);
  return Maybe<void>::Ok();
}

}  // namespace

}  // namespace oneflow
