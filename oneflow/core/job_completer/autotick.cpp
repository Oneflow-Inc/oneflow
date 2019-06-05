#include "oneflow/core/job_completer/autotick.h"
#include "oneflow/core/job/job_builder.h"

namespace oneflow {

namespace {

std::unique_ptr<MutOpConTickInputHelper> NewMutOpConTickInputHelper(const OperatorConf& op_conf) {
  std::unique_ptr<MutOpConTickInputHelper> ret;
  if (IsClassRegistered<MutOpConTickInputHelper>(op_conf.op_type_case())) {
    ret.reset(NewObj<MutOpConTickInputHelper>(op_conf.op_type_case()));
    ret->InitFromOpConf(op_conf);
  }
  return ret;
}

void GroupTickByParallelDesc(const OpGraph& op_graph, Job* job) {
  JobBuilder job_builder(job);
  HashMap<ParallelDesc, std::vector<OpNode*>> parallel_desc2op_node;
  op_graph.ForEachNode([&](OpNode* op_node) {
    auto mut_tick_input_helper = NewMutOpConTickInputHelper(op_node->op().op_conf());
    if (!mut_tick_input_helper) { return; }
    if (mut_tick_input_helper->IsTickInputBound() == true) { return; }
    parallel_desc2op_node[op_node->parallel_desc()].push_back(op_node);
  });
  for (const auto& pair : parallel_desc2op_node) {
    if (pair.first.parallel_num() == 1 && pair.second.size() == 1) { continue; }
    OperatorConf tick_op;
    tick_op.set_name("System-Tick_" + NewUniqueId());
    tick_op.mutable_tick_conf()->set_out("out");
    job_builder.AddOps(pair.first.parallel_conf(), {tick_op});

    for (const auto* op_node : pair.second) {
      auto mut_tick_input_helper = NewMutOpConTickInputHelper(op_node->op().op_conf());
      job_builder.MutOps({mut_tick_input_helper->NewTickInputBoundOpConf(tick_op.name() + "/out")});
    }
  }
}

void ConnectSourceTickAndOtherTick(Job* job) {
  JobBuilder job_builder(job);
  OperatorConf src_tick_op;
  src_tick_op.set_name("System-SourceTick_" + NewUniqueId());
  src_tick_op.mutable_source_tick_conf()->set_out("out");

  job_builder.ForEachOperator([&](const Operator& op) {
    CHECK_EQ(op.op_conf().has_source_tick_conf(), false);
    auto mut_helper = NewMutOpConTickInputHelper(op.op_conf());
    if (!mut_helper) { return; }
    if (mut_helper->IsTickInputBound() == true) { return; }
    job_builder.MutOps({mut_helper->NewTickInputBoundOpConf(src_tick_op.name() + "/out")});
  });

  ParallelConf parallel_conf;
  parallel_conf.set_policy(kDataParallel);
  parallel_conf.add_device_name("0:cpu:0");
  job_builder.AddOps(parallel_conf, {src_tick_op});
}

}  // namespace

void AutoTick(const OpGraph& op_graph, Job* job) {
  GroupTickByParallelDesc(op_graph, job);
  op_graph.ForEachNode([&](OpNode* node) { CHECK(!node->op().op_conf().has_source_tick_conf()); });
  ConnectSourceTickAndOtherTick(job);
}

}  // namespace oneflow
