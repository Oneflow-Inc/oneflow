#include "graph/logical_graph.h"
#include "glog/logging.h"
#include "operator/operator_factory.h"

namespace oneflow {

LogicalGraph::LogicalGraph(const DLNetConf& dl_net_conf,
                           const Strategy& strategy_conf) {
  BuildGraphStruct(dl_net_conf);
  FillNodeWithParallelDesc(strategy_conf);
}

void LogicalGraph::BuildGraphStruct(const DLNetConf& dl_net_conf) {
  HashMap<std::string, LogicalNode*> lbn2node;
  // Process Op
  for (int op_i = 0; op_i < dl_net_conf.op_conf_size(); ++op_i) {
    const OperatorConf& cur_op_conf = dl_net_conf.op_conf(op_i);
    // Construct cur node
    LogicalNode* cur_node = NewFinalNode();
    cur_node->mut_op() = ConstructOpFromPbConf(cur_op_conf);
    // Connect input node
    for (const std::string& ibn : cur_node->op()->input_bns()) {
      std::string lbn = cur_node->op()->ibn2lbn(ibn);
      LogicalNode* pred_node = lbn2node.at(lbn);
      Connect(pred_node, NewFinalEdge(), cur_node);
    }
    // Construct output
    for (const std::string& obn : cur_node->op()->output_bns()) {
      std::string lbn = cur_node->op()->obn2lbn(obn);
      lbn2node.emplace(lbn, cur_node);
    }
  }
  lbn2node.clear();
  // Post Processing
  UpdateSourceAndSink();
}

void LogicalGraph::FillNodeWithParallelDesc(const Strategy& strategy_conf) {
  HashMap<std::string, LogicalNode*> op_name2node;
  for (const std::unique_ptr<LogicalNode>& logical_node : nodes()) {
    const std::string& op_name = logical_node->op()->op_name();
    CHECK(op_name2node.emplace(op_name, logical_node.get()).second);
  }
  for (int gid = 0; gid < strategy_conf.placement_groups_size(); ++gid) {
    const PlacementGroup& cur_group = strategy_conf.placement_groups(gid);
    for (int li = 0; li < cur_group.op_names_size(); ++li) {
      const std::string& op_name = cur_group.op_names(li);
      auto it = op_name2node.find(op_name);
      CHECK(it != op_name2node.end());
      auto parallel_desc_raw_ptr = new ParallelDesc(cur_group.parallel_conf());
      it->second->mut_parallel_desc().reset(parallel_desc_raw_ptr);
    }
  }
}

} // namespace oneflow
