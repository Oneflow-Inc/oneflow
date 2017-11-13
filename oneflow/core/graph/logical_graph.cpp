#include "oneflow/core/graph/logical_graph.h"
#include "oneflow/core/operator/operator_manager.h"

namespace oneflow {

const LogicalNode* LogicalGraph::GetProducerNode(const std::string& lbn) {
  return lbn2producer_.at(lbn);
}

LogicalGraph::LogicalGraph() {
  HashMap<LogicalEdge*, std::string> edge2lbn;
  HashMap<LogicalEdge*, std::string> edge2ibn;
  NaiveBuildGraphStruct(&edge2lbn, &edge2ibn);
  FillNodeWithParallelDesc();
  AddCloneNodes(edge2lbn, edge2ibn);
  ForEachNode([&](LogicalNode* node) {
    for (const std::string& obn : node->op()->output_bns()) {
      const std::string& lbn = node->op()->Lbn4BnInOp(obn);
      CHECK(lbn2producer_.emplace(lbn, node).second);
    }
  });
  ToDotWithAutoFilePath();
}

void LogicalGraph::NaiveBuildGraphStruct(
    HashMap<LogicalEdge*, std::string>* edge2lbn,
    HashMap<LogicalEdge*, std::string>* edge2ibn) {
  const DLNetConf& dlnet_conf = JobDesc::Singleton()->dlnet_conf();
  HashMap<std::string, LogicalNode*> lbn2producer;
  for (const OperatorConf& cur_op_conf : dlnet_conf.op()) {
    LogicalNode* cur_node = NewNode();
    cur_node->mut_op() = OpMgr::Singleton()->AddOp(cur_op_conf);
    for (const std::string& obn : cur_node->op()->output_bns()) {
      const std::string& lbn = cur_node->op()->Lbn4BnInOp(obn);
      CHECK(lbn2producer.emplace(lbn, cur_node).second);
    }
  }
  ForEachNode([&](LogicalNode* cur_node) {
    for (const std::string& ibn : cur_node->op()->input_bns()) {
      const std::string& lbn = cur_node->op()->Lbn4BnInOp(ibn);
      LogicalNode* pred_node = lbn2producer.at(lbn);
      LogicalEdge* edge = NewEdge();
      CHECK(edge2lbn->emplace(edge, lbn).second);
      CHECK(edge2ibn->emplace(edge, ibn).second);
      Connect(pred_node, edge, cur_node);
    }
  });
}

void LogicalGraph::FillNodeWithParallelDesc() {
  const Placement& placement = JobDesc::Singleton()->placement();
  HashMap<std::string, LogicalNode*> op_name2node;
  ForEachNode([&](LogicalNode* logical_node) {
    const std::string& op_name = logical_node->op()->op_name();
    CHECK(op_name2node.emplace(op_name, logical_node).second);
  });
  for (const PlacementGroup& cur_group : placement.placement_group()) {
    for (const std::string& op_name : cur_group.op_set().op_name()) {
      LogicalNode* node = op_name2node.at(op_name);
      auto parallel_desc_raw_ptr = new ParallelDesc(cur_group.parallel_conf());
      node->op()->FixParallelDesc(parallel_desc_raw_ptr);
      node->mut_parallel_desc().reset(parallel_desc_raw_ptr);
    }
  }
  ForEachNode([&](LogicalNode* cur_node) {
    if (cur_node->op()->IsElemWiseOp()) {
      LogicalNode* pred_node = cur_node;
      while (pred_node->op()->IsElemWiseOp()) {
        pred_node = pred_node->SoleInEdge()->src_node();
      }
      cur_node->mut_parallel_desc() = pred_node->parallel_desc();
    }
  });
}

void LogicalGraph::AddCloneNodes(
    const HashMap<LogicalEdge*, std::string>& edge2lbn,
    const HashMap<LogicalEdge*, std::string>& edge2ibn) {
  std::vector<CloneInfo> clone_infos;
  CollectCloneInfos(&clone_infos, edge2lbn);
  for (const CloneInfo& clone_info : clone_infos) {
    AddOneCloneNode(clone_info, edge2ibn);
  }
}

void LogicalGraph::CollectCloneInfos(
    std::vector<CloneInfo>* clone_infos,
    const HashMap<LogicalEdge*, std::string>& edge2lbn) {
  ForEachNode([&](LogicalNode* cur_node) {
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
      pb_op_conf.set_name("clone_" + lbn);
      pb_op_conf.mutable_clone_conf()->set_out_num(edges.size());
      pb_op_conf.mutable_clone_conf()->set_lbn(lbn);
      auto clone_op = OpMgr::Singleton()->AddOp(pb_op_conf);
      // Set clone_info
      CloneInfo clone_info;
      clone_info.clone_op = clone_op;
      clone_info.pred_node = cur_node;
      clone_info.edges = std::move(edges);
      clone_infos->push_back(clone_info);
    }
  });
}

void LogicalGraph::AddOneCloneNode(
    const CloneInfo& clone_info,
    const HashMap<LogicalEdge*, std::string>& edge2ibn) {
  if (clone_info.pred_node->op()->IsDataLoaderOp()) { return; }
  LogicalNode* clone_node = NewNode();
  clone_node->mut_op() = clone_info.clone_op;
  clone_node->mut_parallel_desc() = clone_info.pred_node->parallel_desc();
  Connect(clone_info.pred_node, NewEdge(), clone_node);
  CHECK_EQ(clone_node->op()->output_bns().size(), clone_info.edges.size());
  for (size_t i = 0; i < clone_info.edges.size(); ++i) {
    const std::string& obn = clone_node->op()->output_bns().at(i);
    const std::string& lbn = clone_node->op()->Lbn4BnInOp(obn);
    LogicalEdge* edge = clone_info.edges.at(i);
    const std::string& ibn = edge2ibn.at(edge);
    LogicalNode* dst_node = edge->dst_node();
    dst_node->mut_op()->ModifyLbn4BnInOp(ibn, lbn);
    dst_node->mut_op()->TryModifyLbn4BnInOp(GenDiffBn(ibn), lbn);
    DisConnect(edge);
    Connect(clone_node, edge, dst_node);
  }
}

}  // namespace oneflow
