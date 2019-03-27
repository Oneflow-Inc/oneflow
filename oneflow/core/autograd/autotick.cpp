#include "oneflow/core/autograd/autotick.h"
#include "oneflow/core/job/job_conf_builder.h"

namespace oneflow {

namespace {

void AddAutoTick4VariableOp(const OpGraph& op_graph, JobConf1* job_conf) {
  JobConfBuilder job_conf_builder(job_conf);
  HashMap<ParallelDesc, std::vector<OpNode*>> parallel_desc2op_node;
  op_graph.ForEachNode([&](OpNode* op_node) {
    if (!op_node->op().op_conf().has_variable_conf()) { return; }
    const auto& conf = op_node->op().op_conf().variable_conf();
    if (conf.has_tick()) { return; }
    parallel_desc2op_node[op_node->parallel_desc()].push_back(op_node);
  });
  for (const auto& pair : parallel_desc2op_node) {
    OperatorConf tick_op;
    tick_op.set_name("tick_" + NewUniqueId());
    tick_op.mutable_tick_conf()->set_out("out");
    job_conf_builder.AddOps(pair.first.parallel_conf(), {tick_op});

    std::vector<OperatorConf> var_ops;
    for (const auto* var_op_node : pair.second) {
      OperatorConf new_var_op(var_op_node->op().op_conf());
      new_var_op.mutable_variable_conf()->set_tick(tick_op.name() + "/out");
      var_ops.push_back(new_var_op);
    }
    job_conf_builder.MutOps(var_ops);
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

void ConnectSrcAndTickIfNeed(const OpGraph& op_graph, const OpNode& src_node, JobConf1* job_conf) {
  JobConfBuilder job_conf_builder(job_conf);
  std::vector<const OperatorConf*> tick_ops;
  op_graph.ForEachNode([&](OpNode* op_node) {
    if (!op_node->op().op_conf().has_tick_conf()) { return; }
    const auto& conf = op_node->op().op_conf().tick_conf();
    if (conf.has_in()) { return; }
    tick_ops.push_back(&op_node->op().op_conf());
  });
  if (tick_ops.empty()) { return; }
  OperatorConf src_tick_op;
  if (tick_ops.size() == 1) {
    src_tick_op = *tick_ops.at(0);
  } else {
    src_tick_op.set_name("tick_" + NewUniqueId());
    src_tick_op.mutable_tick_conf()->set_out("out");
    std::vector<OperatorConf> mut_ops;
    for (const auto* dst_tick_op : tick_ops) {
      OperatorConf mut_tick_op(*dst_tick_op);
      mut_tick_op.mutable_tick_conf()->set_in(src_tick_op.name() + "/out");
      mut_ops.push_back(mut_tick_op);
    }
    job_conf_builder.MutOps(mut_ops);
  }
  src_tick_op.mutable_tick_conf()->set_in(
      GenLogicalBlobName(src_node.op().BnInOp2Lbi(src_node.op().output_bns().Get(0))));
  job_conf_builder.AddOrMutOps(src_node.parallel_desc().parallel_conf(), {src_tick_op});
}

}  // namespace

void AutoTick(const OpGraph& op_graph, JobConf1* job_conf) {
  AddAutoTick4VariableOp(op_graph, job_conf);
  OpNode* src_op_node = FindSourceOpNode(op_graph);
  if (src_op_node != nullptr) { ConnectSrcAndTickIfNeed(op_graph, *src_op_node, job_conf); }
}

}  // namespace oneflow
