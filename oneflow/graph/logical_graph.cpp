#include "graph/logical_graph.h"
#include "glog/logging.h"
#include "operator/operator_factory.h"

namespace oneflow {

void LogicalGraph::Init(const DLNetConf& dl_net_conf,
                      const Strategy& strategy_conf) {
  Graph::Init();
  BuildGraphStruct(dl_net_conf);
  FillNodeWithParallelDesc(strategy_conf);
}

void LogicalGraph::BuildGraphStruct(const DLNetConf& dl_net_conf) {
  std::unordered_map<std::string, LogicalNode*> lbn2node;
  // Process Op
  for (int op_i = 0; op_i < dl_net_conf.op_conf_size(); ++op_i) {
    const OperatorConf& cur_op_conf = dl_net_conf.op_conf(op_i);
    // Construct cur node
    LogicalNode* cur_node = NewFinalNode();
    cur_node->mutable_op_ptr() = ConstructOpFromPbConf(cur_op_conf);
    // Connect input node
    for (const std::string& ibn : cur_node->op().input_blob_names()) {
      std::string lbn = cur_node->op().ibn2lbn(ibn);
      LogicalNode* pred_node = lbn2node.at(lbn);
      Connect(pred_node, NewFinalEdge(), cur_node);
    }
    // Construct output
    for (const std::string& obn : cur_node->op().output_blob_names()) {
      std::string lbn = cur_node->op().obn2lbn(obn);
      lbn2node.emplace(lbn, cur_node);
    }
  }
  lbn2node.clear();
  // Post Processing
  UpdateStartAndStop();
}

void LogicalGraph::FillNodeWithParallelDesc(const Strategy& strategy_conf) {
  std::unordered_map<std::string, LogicalNode*> op_name2node;
  for (const std::unique_ptr<LogicalNode>& logical_node : nodes()) {
    std::string op_name = logical_node->op().op_name();
    bool emplace_success =
      op_name2node.emplace(op_name, logical_node.get()).second;
    CHECK_EQ(emplace_success, true);
  }
  for (int gid = 0; gid < strategy_conf.placement_group_vec_size(); ++gid) {
    const PlacementGroup& cur_group = strategy_conf.placement_group_vec(gid);
    for (int li = 0; li < cur_group.op_name_vec_size(); ++li) {
      const std::string& op_name = cur_group.op_name_vec(li);
      auto it = op_name2node.find(op_name);
      CHECK(it != op_name2node.end());
      ParallelDesc* parallel_desc_raw_ptr = new ParallelDesc;
      parallel_desc_raw_ptr->Init(cur_group.parallel_conf());
      it->second->mutable_parallel_desc_ptr().reset(parallel_desc_raw_ptr);
    }
  }
}

} // namespace oneflow
