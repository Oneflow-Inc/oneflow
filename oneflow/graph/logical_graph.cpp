#include "graph/logical_graph.h"
#include "glog/logging.h"
#include "operator/operator_factory.h"

namespace oneflow {

LogicalGraph::LogicalGraph(const DLNetConf& dl_net_conf,
                           const Strategy& strategy_conf) {
  HashMap<LogicalEdge*, std::string> edge2lbn;
  NaiveBuildGraphStruct(dl_net_conf, &edge2lbn);
  FillNodeWithParallelDesc(strategy_conf);
  AddCloneNodes(edge2lbn);
}

void LogicalGraph::NaiveBuildGraphStruct(
    const DLNetConf& dl_net_conf,
    HashMap<LogicalEdge*, std::string>* edge2lbn) {
  HashMap<std::string, LogicalNode*> lbn2producer;
  // Process Op
  for (int op_i = 0; op_i < dl_net_conf.op_conf_size(); ++op_i) {
    const OperatorConf& cur_op_conf = dl_net_conf.op_conf(op_i);
    // Construct cur node
    LogicalNode* cur_node = NewFinalNode();
    cur_node->mut_op() = ConstructOpFromPbConf(cur_op_conf);
    // Connect input node
    for (const std::string& ibn : cur_node->op()->input_bns()) {
      std::string lbn = cur_node->op()->ibn2lbn(ibn);
      LogicalNode* pred_node = lbn2producer.at(lbn);
      LogicalEdge* edge = NewFinalEdge();
      CHECK(edge2lbn->emplace(edge, lbn).second);
      Connect(pred_node, edge, cur_node);
    }
    // Construct output
    for (const std::string& obn : cur_node->op()->output_bns()) {
      std::string lbn = cur_node->op()->obn2lbn(obn);
      lbn2producer.emplace(lbn, cur_node);
    }
  }
  lbn2producer.clear();
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

void LogicalGraph::AddCloneNodes(
    const HashMap<LogicalEdge*, std::string>& edge2lbn) {
  std::vector<CloneInfo> clone_infos;
  CollectCloneInfos(&clone_infos, edge2lbn);
  for (const CloneInfo& clone_info : clone_infos) {
    AddOneCloneNode(clone_info);
  }
}

void LogicalGraph::CollectCloneInfos(
    std::vector<CloneInfo>* clone_infos,
    const HashMap<LogicalEdge*, std::string>& edge2lbn) {
  for (const std::unique_ptr<LogicalNode>& cur_node : nodes()) {
    HashMap<std::string, std::vector<LogicalEdge*>> lbn2edges;
    for (LogicalEdge* edge : cur_node->out_edges()) {
      lbn2edges[edge2lbn.at(edge)].push_back(edge);
    }
    for (auto& pair : lbn2edges) {
      const std::string& lbn = pair.first;
      std::vector<LogicalEdge*>& edges = pair.second;
      if (edges.size() <= 1) { continue; }
      // Construct clone op
      OperatorConf pb_op_conf;
      pb_op_conf.set_name("clone_" + lbn + "_" + cur_node->node_id_str());
      pb_op_conf.mutable_clone_op_conf()->set_out_num(edges.size());
      pb_op_conf.mutable_clone_op_conf()->set_lbn(lbn);
      pb_op_conf.mutable_clone_op_conf()->set_is_boxing(false);
      auto clone_op = ConstructOpFromPbConf(pb_op_conf);
      // Set clone_info
      CloneInfo clone_info;
      clone_info.clone_op = clone_op;
      clone_info.pred_node = cur_node.get();
      clone_info.edges = std::move(edges);
      clone_infos->push_back(clone_info);
    }
  }
}

void LogicalGraph::AddOneCloneNode(const CloneInfo& clone_info) {
  LogicalNode* clone_node = NewFinalNode();
  clone_node->mut_op() = clone_info.clone_op;
  Connect(clone_info.pred_node, NewFinalEdge(), clone_node);
  CHECK_EQ(clone_node->op()->output_bns().size(), clone_info.edges.size());
  for (LogicalEdge* edge : clone_info.edges) {
    LogicalNode* dst_node = edge->dst_node();
    DisConnect(edge);
    Connect(clone_node, edge, dst_node);
  }
}

} // namespace oneflow
