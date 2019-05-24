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

void AddAutoTickOpConf(const OpGraph& op_graph, Job* job) {
  JobBuilder job_builder(job);
  HashMap<ParallelDesc, std::vector<OpNode*>> parallel_desc2op_node;
  op_graph.ForEachNode([&](OpNode* op_node) {
    auto mut_tick_input_helper = NewMutOpConTickInputHelper(op_node->op().op_conf());
    if (!mut_tick_input_helper) { return; }
    if (mut_tick_input_helper->IsTickInputBound() == true) { return; }
    parallel_desc2op_node[op_node->parallel_desc()].push_back(op_node);
  });
  for (const auto& pair : parallel_desc2op_node) {
    OperatorConf tick_op;
    tick_op.set_name("tick_" + NewUniqueId());
    tick_op.mutable_tick_conf()->set_out("out");
    job_builder.AddOps(pair.first.parallel_conf(), {tick_op});

    std::vector<OperatorConf> var_ops;
    for (const auto* op_node : pair.second) {
      auto mut_tick_input_helper = NewMutOpConTickInputHelper(op_node->op().op_conf());
      var_ops.push_back(mut_tick_input_helper->NewTickInputBoundOpConf(tick_op.name() + "/out"));
    }
    job_builder.MutOps(var_ops);
  }
}

OpNode* FindSourceOpNode(const OpGraph& op_graph) {
  OpNode* ret = nullptr;
  op_graph.ForEachNode([&](OpNode* op_node) {
    if (ret != nullptr) { return; }
    if (op_node->out_edges().empty()) { return; }
    if (op_node->op().op_conf().has_decode_ofrecord_conf()) {
      ret = op_node;
    } else if (op_node->op().op_conf().has_decode_random_conf()) {
      ret = op_node;
    } else if (op_node->op().op_conf().has_define_test_blob_conf()) {
      ret = op_node;
    } else {
      // do nothing
    }
  });
  return ret;
}

void ConnectSrcAndTickIfNeed(const OpNode& src_node, Job* job) {
  JobBuilder job_builder(job);
  std::vector<const OperatorConf*> no_in_tick_ops;
  for (const OperatorConf& op_conf : job->net().op()) {
    if (!op_conf.has_tick_conf()) { continue; }
    if (op_conf.tick_conf().has_in()) { continue; }
    no_in_tick_ops.push_back(&op_conf);
  }
  if (no_in_tick_ops.empty()) { return; }
  OperatorConf src_tick_op;
  if (no_in_tick_ops.size() == 1) {
    src_tick_op = *no_in_tick_ops.at(0);
  } else {
    src_tick_op.set_name("tick_" + NewUniqueId());
    src_tick_op.mutable_tick_conf()->set_out("out");
    std::vector<OperatorConf> mut_ops;
    for (const auto* dst_tick_op : no_in_tick_ops) {
      OperatorConf mut_tick_op(*dst_tick_op);
      mut_tick_op.mutable_tick_conf()->set_in(src_tick_op.name() + "/out");
      mut_ops.push_back(mut_tick_op);
    }
    job_builder.MutOps(mut_ops);
  }
  src_tick_op.mutable_tick_conf()->set_in(
      GenLogicalBlobName(src_node.op().BnInOp2Lbi(src_node.op().output_bns().Get(0))));
  job_builder.AddOrMutOps(src_node.parallel_desc().parallel_conf(), {src_tick_op});
}

}  // namespace

void AutoTick(const OpGraph& op_graph, Job* job) {
  AddAutoTickOpConf(op_graph, job);
  OpNode* src_op_node = FindSourceOpNode(op_graph);
  if (src_op_node != nullptr) { ConnectSrcAndTickIfNeed(*src_op_node, job); }
}

}  // namespace oneflow
