#include "oneflow/core/job_completer/job_completer.h"
#include "oneflow/core/job_completer/op_graph_pass.h"
#include "oneflow/core/job_completer/autograd.h"
#include "oneflow/core/job_completer/autotick.h"
#include "oneflow/core/job_completer/add_keep_header_only_op_conf.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job_completer/group_boxing_by_dst_parallel.h"
#include "oneflow/core/framework/config_def.h"
#include "oneflow/core/job_completer/xrt_compilation.h"

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
      mut_mutable_consumer_op_conf.add_ctrl_in_op_name(fw_bw_op_name);
    }
    job_builder->MutOpsOnlyOnce({mut_mutable_consumer_op_conf});
  }
}

void SetCtrlInOp4NVTXOp(const OpGraph& op_graph, JobBuilder* job_builder) {
  auto IsNVTXOp = [](const OperatorConf& op_conf) -> bool {
    return op_conf.has_nvtx_range_start_conf() || op_conf.has_nvtx_range_end_conf();
  };
  HashMap<std::string, OperatorConf> op_name2consumer_op_conf{};
  auto IsReachable = op_graph.MakePredicatorIsLbiAllConsumersReachableToOpName();
  op_graph.ForEachNode([&](OpNode* op_node) {
    if (IsNVTXOp(op_node->op().op_conf()) == false) { return; }
    const Operator& nvtx_op = op_node->op();
    HashSet<std::string> op_name_of_controlled{};
    for (OpEdge* in_edge_of_nvtx : op_node->in_edges()) {
      for (OpEdge* edge : in_edge_of_nvtx->src_node()->out_edges()) {
        bool will_create_circle = false;
        for (auto& lbi : edge->lbis()) {
          if (IsReachable(lbi, nvtx_op.op_name())) {
            CHECK(will_create_circle == false);
            will_create_circle = true;
            continue;
          }
          // TODO: this check could be redundant since ctrl in of one op won't be duplicated.
          for (const std::string controlled : op_name_of_controlled) {
            if (IsReachable(lbi, controlled)) {
              CHECK(will_create_circle == false);
              will_create_circle = true;
              continue;
            }
          }
        }
        const OperatorConf& consumer_op_conf = edge->dst_node()->op().op_conf();
        for (OpEdge* in_edge_of_nvtx : op_node->in_edges()) {
          if (will_create_circle == false
              && in_edge_of_nvtx->src_node()->op().op_name() == consumer_op_conf.name()) {
            will_create_circle = true;
            continue;
          }
        }
        if (IsNVTXOp(consumer_op_conf) || will_create_circle) { continue; }
        if (consumer_op_conf.name() == nvtx_op.op_name()) { continue; }
        auto iter = op_name2consumer_op_conf.find(consumer_op_conf.name());
        if (iter == op_name2consumer_op_conf.end()) {
          OperatorConf mut_consumer_op_conf(consumer_op_conf);
          mut_consumer_op_conf.add_ctrl_in_op_name(nvtx_op.op_name());
          op_name2consumer_op_conf.emplace(consumer_op_conf.name(), mut_consumer_op_conf);
        } else {
          const auto& existed_ctrl_in_op_names = iter->second.ctrl_in_op_name();
          if (std::find(existed_ctrl_in_op_names.begin(), existed_ctrl_in_op_names.end(),
                        nvtx_op.op_name())
              == existed_ctrl_in_op_names.end()) {
            iter->second.add_ctrl_in_op_name(nvtx_op.op_name());
          }
        }
        op_name_of_controlled.emplace(consumer_op_conf.name());
      }
    }
  });
  for (auto& pair : op_name2consumer_op_conf) { job_builder->MutOpsOnlyOnce({pair.second}); }
}

}  // namespace

void JobCompleter::Complete(Job* job) const {
  FunctionPass("DumpTimeShapeAndBlobParallelConfPass")(job);
  WithOpGraphAndMutJobBuilder(job, &GroupBoxingByDstParallel);
  WithOpGraphAndMutJobBuilder(job, &AddKeepHeaderOnlyOp);
  WithOpGraphAndMutJobBuilder(job, &SetCtrlInOpName4VariableOp);
  WithOpGraphAndMutJobBuilder(job, &SetCtrlInOp4NVTXOp);
  // complete tick ops
  WithOpGraphAndMutJobBuilder(job, &AutoSourceTick);
  WithOpGraphAndMutJobBuilder(job, &AddTickForTimeShape);
  WithOpGraphAndMutJobBuilder(job, &AutoSinkTick);
  AddGlobalTotalJobCriticalSection(*job);
  WithOpGraphAndMutJobBuilder(job, &AddGlobalInputCriticalSections);
  WithOpGraphAndMutJobBuilder(job, &AddGlobalOutputCriticalSections);
  FunctionPass("DumpTimeShapeAndBlobParallelConfPass")(job);
  if (XrtCompilationEnabled(GlobalJobDesc())) {
#ifdef OF_WITH_XRT
    WithOpGraphAndMutJob(job, &RebuildXrtCompiledJob);
#else
    LOG(WARNING) << "It will not use XLA or TensorRT since WITH_XLA or "
                    "WITH_TENSORRT was not enabled when compiling the project.";
#endif  // OF_WITH_XRT
  }
  CheckOpGraph(OpGraph(*job));
}

}  // namespace oneflow
