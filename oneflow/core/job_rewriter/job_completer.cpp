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
#include "oneflow/core/job_rewriter/job_completer.h"
#include "oneflow/core/job_rewriter/job_pass.h"
#include "oneflow/core/job_rewriter/autograd.h"
#include "oneflow/core/job_rewriter/autotick.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job_rewriter/group_boxing_by_dst_parallel.h"
#include "oneflow/core/framework/config_def.h"
#include "oneflow/core/job_rewriter/boxing_with_middle_nodes.h"

namespace oneflow {

namespace {

Maybe<void> CheckOpGraph(const OpGraph& op_graph) {
  JUST(op_graph.MaybeForEachNode([&](OpNode* op_node) -> Maybe<void> {
    size_t in_cnt = 0;
    op_graph.ForEachDataAndCtrlInNode(op_node, [&](OpNode*) { ++in_cnt; });
    if (in_cnt == 0) { CHECK_OR_RETURN(op_node->op().op_conf().has_wait_and_send_ids_conf()); }

    size_t out_cnt = 0;
    op_graph.ForEachDataAndCtrlOutNode(op_node, [&](OpNode*) { ++out_cnt; });

    if (out_cnt == 0) { CHECK_OR_RETURN(op_node->op().op_conf().has_callback_notify_conf()); }
    return Maybe<void>::Ok();
  }));
  return Maybe<void>::Ok();
}

Maybe<void> WithOpGraphAndMutJob(Job* job,
                                 const std::function<Maybe<void>(const OpGraph&, Job*)>& Handler) {
  OpGraph op_graph(*job);
  JUST(Handler(op_graph, job));
  return Maybe<void>::Ok();
}

Maybe<void> WithOpGraphAndMutJobBuilder(
    Job* job, const std::function<Maybe<void>(const OpGraph&, JobBuilder*)>& Handler) {
  OpGraph op_graph(*job);
  JobBuilder job_builder(job);
  JUST(Handler(op_graph, &job_builder));
  return Maybe<void>::Ok();
}

Maybe<void> SetCtrlInOpName4VariableOp(const OpGraph& op_graph, JobBuilder* job_builder) {
  auto IsMutableConsumedLbi = [](const Operator& op, const LogicalBlobId& lbi) -> bool {
    for (const std::string& bn : op.input_bns()) {
      if (op.BnInOp2Lbi(bn) == lbi && op.InputBlobModifier4Ibn(bn).is_mutable()) { return true; }
    }
    return false;
  };
  auto IsReachable = op_graph.MakePredicatorIsOpNameDataOrCtrlReachable();
  HashMap<const OperatorConf*, HashSet<std::string>> op_conf2ctrl_in_op_names;
  JUST(op_graph.MaybeForEachNode([&](OpNode* op_node) -> Maybe<void> {
    if (op_node->op().op_conf().has_variable_conf() == false) { return Maybe<void>::Ok(); }
    if (op_node->out_edges().size() <= 1) { return Maybe<void>::Ok(); }
    const Operator& variable_op = op_node->op();
    const LogicalBlobId& variable_lbi = variable_op.BnInOp2Lbi(variable_op.SoleObn());
    const OperatorConf* mutable_consumer = nullptr;
    std::vector<const OperatorConf*> naive_consumers;
    naive_consumers.reserve(op_node->out_edges().size());
    for (OpEdge* edge : op_node->out_edges()) {
      const auto& op_conf = edge->dst_node()->op().op_conf();
      if (IsMutableConsumedLbi(edge->dst_node()->op(), variable_lbi)) {
        CHECK_OR_RETURN(mutable_consumer == nullptr);
        mutable_consumer = &op_conf;
      } else {
        naive_consumers.emplace_back(&op_conf);
      }
    }
    if (mutable_consumer == nullptr) { return Maybe<void>::Ok(); }
    for (const auto* fw_bw_op : naive_consumers) {
      op_conf2ctrl_in_op_names[mutable_consumer].insert(fw_bw_op->name());
    }
    return Maybe<void>::Ok();
  }));
  for (const auto& pair : op_conf2ctrl_in_op_names) {
    OperatorConf mut_mutable_consumer_op_conf(*pair.first);
    for (const auto& fw_bw_op_name : pair.second) {
      if (!IsReachable(fw_bw_op_name, mut_mutable_consumer_op_conf.name())) {
        mut_mutable_consumer_op_conf.add_ctrl_in_op_name(fw_bw_op_name);
      }
    }
    JUST(job_builder->MutOpOnlyOnce(mut_mutable_consumer_op_conf));
  }
  return Maybe<void>::Ok();
}

}  // namespace

Maybe<void> JobCompleter::Complete(Job* job) const {
  JobPassCtx job_pass_ctx(GlobalJobDesc());
  JUST(JobPass4Name("DumpBlobParallelConfPass")(job, &job_pass_ctx));
  // NOTE(chengcheng): disable this pass for reduce boxing memory life cycle to memory cost.
  if (!Global<ResourceDesc, ForSession>::Get()->resource().disable_group_boxing_by_dst_parallel()) {
    JUST(WithOpGraphAndMutJobBuilder(job, &GroupBoxingByDstParallel));
  }
  JUST(WithOpGraphAndMutJobBuilder(job, &BoxingWithMiddleNodes));
  JUST(WithOpGraphAndMutJobBuilder(job, &SetCtrlInOpName4VariableOp));
  // complete tick ops
  JUST(WithOpGraphAndMutJobBuilder(job, &AutoPrependTick));
  JUST(WithOpGraphAndMutJobBuilder(job, &AddTickForTimeShape));
  JUST(WithOpGraphAndMutJob(job, &MultiClientAutoSourceAndSinkTick));
  JUST(WithOpGraphAndMutJob(job, &MultiClientAutoInterfaceCriticalSectionTick));
  JUST(JobPass4Name("SystemOpFillJobNamePass")(job, &job_pass_ctx));
  JUST(JobPass4Name("DumpBlobParallelConfPass")(job, &job_pass_ctx));
#ifdef WITH_CUDA
  if (Global<ResourceDesc, ForSession>::Get()->nccl_use_compute_stream()) {
    // NOTE(chengcheng): this pass need as last pass for insert correct op with nccl boxing.
    JUST(JobPass4Name("InsertNcclLogicalOpPass")(job, &job_pass_ctx));
    // NOTE(chengcheng): Becasue insert new logical nccl op, MUST dump time shape, sbp again.
    JUST(JobPass4Name("DumpBlobParallelConfPass")(job, &job_pass_ctx));
  }
#endif  // WITH_CUDA
  JUST(CheckOpGraph(OpGraph(*job)));
  return Maybe<void>::Ok();
}

}  // namespace oneflow
