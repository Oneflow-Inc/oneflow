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
#include "oneflow/core/job_rewriter/xrt_compilation.h"

namespace oneflow {

namespace {

void CheckOpGraph(const OpGraph& op_graph) {
  op_graph.ForEachNode([&](OpNode* op_node) {
    size_t in_cnt = 0;
    op_graph.ForEachDataAndCtrlInNode(op_node, [&](OpNode*) { ++in_cnt; });
    if (in_cnt == 0) { CHECK(op_node->op().op_conf().has_source_tick_conf()); }
    size_t out_cnt = 0;
    op_graph.ForEachDataAndCtrlOutNode(op_node, [&](OpNode*) { ++out_cnt; });
    if (out_cnt == 0) { CHECK(op_node->op().op_conf().has_sink_tick_conf()); }
  });
}

void WithOpGraphAndMutJob(Job* job, const std::function<void(const OpGraph&, Job*)>& Handler) {
  OpGraph op_graph(*job);
  Handler(op_graph, job);
}

void WithOpGraphAndMutJobBuilder(Job* job,
                                 const std::function<void(const OpGraph&, JobBuilder*)>& Handler) {
  OpGraph op_graph(*job);
  JobBuilder job_builder(job);
  Handler(op_graph, &job_builder);
}

void SetCtrlInOpName4VariableOp(const OpGraph& op_graph, JobBuilder* job_builder) {
  auto IsMutableConsumedLbi = [](const Operator& op, const LogicalBlobId& lbi) -> bool {
    for (const std::string& bn : op.input_bns()) {
      if (op.BnInOp2Lbi(bn) == lbi && op.InputBlobModifier4Ibn(bn).is_mutable()) { return true; }
    }
    return false;
  };
  auto IsReachable = op_graph.MakePredicatorIsOpNameDataOrCtrlReachable();
  HashMap<const OperatorConf*, HashSet<std::string>> op_conf2ctrl_in_op_names;
  op_graph.ForEachNode([&](OpNode* op_node) {
    if (op_node->op().op_conf().has_variable_conf() == false) { return; }
    if (op_node->out_edges().size() <= 1) { return; }
    const Operator& variable_op = op_node->op();
    const LogicalBlobId& variable_lbi = variable_op.BnInOp2Lbi(variable_op.SoleObn());
    const OperatorConf* mutable_consumer = nullptr;
    std::vector<const OperatorConf*> naive_consumers;
    for (OpEdge* edge : op_node->out_edges()) {
      const auto& op_conf = edge->dst_node()->op().op_conf();
      if (IsMutableConsumedLbi(edge->dst_node()->op(), variable_lbi)) {
        CHECK(mutable_consumer == nullptr);
        mutable_consumer = &op_conf;
      } else {
        naive_consumers.push_back(&op_conf);
      }
    }
    if (mutable_consumer == nullptr) { return; }
    for (const auto* fw_bw_op : naive_consumers) {
      op_conf2ctrl_in_op_names[mutable_consumer].insert(fw_bw_op->name());
    }
  });
  for (const auto& pair : op_conf2ctrl_in_op_names) {
    OperatorConf mut_mutable_consumer_op_conf(*pair.first);
    for (const auto& fw_bw_op_name : pair.second) {
      if (!IsReachable(fw_bw_op_name, mut_mutable_consumer_op_conf.name())) {
        mut_mutable_consumer_op_conf.add_ctrl_in_op_name(fw_bw_op_name);
      }
    }
    job_builder->MutOpsOnlyOnce({mut_mutable_consumer_op_conf});
  }
}

}  // namespace

void JobCompleter::Complete(Job* job) const {
  JobPassCtx job_pass_ctx(GlobalJobDesc());
  JobPass4Name("DumpTimeShapeAndBlobParallelConfPass")(job, &job_pass_ctx);
  // NOTE(chengcheng): disable this pass for reduce boxing memory life cycle to memory cost.
  if (!Global<ResourceDesc, ForSession>::Get()->resource().disable_group_boxing_by_dst_parallel()) {
    WithOpGraphAndMutJobBuilder(job, &GroupBoxingByDstParallel);
  }
  WithOpGraphAndMutJobBuilder(job, &SetCtrlInOpName4VariableOp);
  // complete tick ops
  WithOpGraphAndMutJobBuilder(job, &AutoPrependTick);
  WithOpGraphAndMutJobBuilder(job, &AddTickForTimeShape);
  WithOpGraphAndMutJobBuilder(job, &AutoSourceAndSinkTick);
  WithOpGraphAndMutJobBuilder(job, &AddGlobalInputCriticalSections);
  WithOpGraphAndMutJobBuilder(job, &AddGlobalOutputCriticalSections);
  JobPass4Name("DumpTimeShapeAndBlobParallelConfPass")(job, &job_pass_ctx);
  if (XrtCompilationEnabled(GlobalJobDesc())) {
#ifdef OF_WITH_XRT
    WithOpGraphAndMutJob(job, &RebuildXrtCompiledJob);
#else
    LOG(WARNING) << "It will not use XLA or TensorRT since WITH_XLA or "
                    "WITH_TENSORRT was not enabled when compiling the project.";
#endif  // OF_WITH_XRT
  }

  if (Global<ResourceDesc, ForSession>::Get()->nccl_use_compute_stream()) {
    // NOTE(chengcheng): this pass need as last pass for insert correct op with nccl boxing.
    JobPass4Name("InsertNcclLogicalOpPass")(job, &job_pass_ctx);
    // NOTE(chengcheng): Becasue insert new logical nccl op, MUST dump time shape, sbp again.
    JobPass4Name("DumpTimeShapeAndBlobParallelConfPass")(job, &job_pass_ctx);
  }
  CheckOpGraph(OpGraph(*job));
}

}  // namespace oneflow
