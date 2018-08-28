#include "oneflow/core/operator/fully_connected_op.h"
#include "oneflow/core/graph/reduce_graph.h"
#include "oneflow/core/graph/logical_graph.h"

namespace oneflow {

struct ReduceGraph::Group {
  std::vector<const LogicalNode*> nodes;
  HashSet<const LogicalNode*> ancestors;
  HashSet<const LogicalNode*> ancestors_and_this;
  HashSet<const LogicalNode*> descendants;
  HashSet<const LogicalNode*> descendants_and_this;
  bool is_mergeable;

  bool IsMergeable() const { return is_mergeable; };

  bool IsParallelDescEqual(const Group& rhs) const {
    CHECK_GT(nodes.size(), 0);
    CHECK_GT(rhs.nodes.size(), 0);
    return nodes.front()->parallel_desc()->Equal(rhs.nodes.front()->parallel_desc().get());
  }
};

ReduceGraph::ReduceGraph(const LogicalGraph& logical_graph) {
  std::list<Group> group_list;
  HashMap<const LogicalNode*, std::list<Group>::iterator> logical2group_it;

  InitGroups(logical_graph, &group_list, &logical2group_it);
  MergeGroups(&group_list, &logical2group_it);
  BuildGraph(logical_graph, &group_list);
}

void ReduceGraph::InitGroups(
    const LogicalGraph& logical_graph, std::list<Group>* group_list,
    HashMap<const LogicalNode*, std::list<Group>::iterator>* logical2group_it) {
  logical_graph.ForEachNode([&](const LogicalNode* node) {
    group_list->emplace_back();
    logical2group_it->insert({node, --group_list->end()});
    Group& group = group_list->back();
    group.nodes = {node};
    group.is_mergeable = IsLogicalNodeMergeable(node);
  });

  logical_graph.TopoForEachNode([&](const LogicalNode* node) {
    auto cur_group = logical2group_it->at(node);

    for (const LogicalEdge* edge : node->in_edges()) {
      LogicalNode* pred_node = edge->src_node();
      auto pred_group = logical2group_it->at(pred_node);
      cur_group->ancestors.insert(pred_group->ancestors.begin(), pred_group->ancestors.end());
      cur_group->ancestors.insert(pred_node);
    }
    cur_group->ancestors_and_this.insert(cur_group->ancestors.begin(), cur_group->ancestors.end());
    cur_group->ancestors_and_this.insert(cur_group->nodes.begin(), cur_group->nodes.end());
  });

  logical_graph.ReverseTopoForEachNode([&](const LogicalNode* node) {
    auto cur_group = logical2group_it->at(node);

    for (const LogicalEdge* edge : node->out_edges()) {
      LogicalNode* succ_node = edge->dst_node();
      auto succ_group = logical2group_it->at(succ_node);
      cur_group->descendants.insert(succ_group->descendants.begin(), succ_group->descendants.end());
      cur_group->descendants.insert(succ_node);
    }

    cur_group->descendants_and_this.insert(cur_group->descendants.begin(),
                                           cur_group->descendants.end());
    cur_group->descendants_and_this.insert(cur_group->nodes.begin(), cur_group->nodes.end());
  });
}

void ReduceGraph::MergeGroups(
    std::list<Group>* group_list,
    HashMap<const LogicalNode*, std::list<Group>::iterator>* logical2group_it) {
  while (group_list->size() > 1 && TryMergeOneGroup(group_list, logical2group_it)) {};
}

bool ReduceGraph::TryMergeOneGroup(
    std::list<Group>* group_list,
    HashMap<const LogicalNode*, std::list<Group>::iterator>* logical2group_it) {
  for (auto lhs = group_list->begin(); lhs != group_list->end(); ++lhs) {
    if (!lhs->IsMergeable()) { continue; }
    for (auto rhs = lhs; rhs != group_list->end(); ++rhs) {
      if (lhs == rhs) { continue; }
      if (!rhs->IsMergeable()) { continue; }
      if (!lhs->IsParallelDescEqual(*rhs)) { continue; }
      if (lhs->ancestors != rhs->ancestors || lhs->descendants != rhs->descendants) { continue; }
      for (const LogicalNode* node : rhs->nodes) {
        lhs->nodes.push_back(node);
        lhs->ancestors_and_this.insert(node);
        lhs->descendants_and_this.insert(node);
        logical2group_it->at(node) = lhs;
      }
      group_list->erase(rhs);
      return true;
    }
  }

  for (auto succ_group_it = group_list->begin(); succ_group_it != group_list->end();
       ++succ_group_it) {
    if (!succ_group_it->IsMergeable()) { continue; }

    for (const LogicalNode* node_in_succ : succ_group_it->nodes) {
      for (const LogicalEdge* in_edge : node_in_succ->in_edges()) {
        auto pred_group_it = logical2group_it->at(in_edge->src_node());
        if (pred_group_it == succ_group_it) { continue; }
        if (!pred_group_it->IsMergeable()) { continue; }
        if (!pred_group_it->IsParallelDescEqual(*succ_group_it)) { continue; }
        if (pred_group_it->ancestors_and_this != succ_group_it->ancestors
            || pred_group_it->descendants != succ_group_it->descendants_and_this) {
          continue;
        }

        for (const LogicalNode* node : succ_group_it->nodes) {
          pred_group_it->nodes.push_back(node);
          pred_group_it->ancestors_and_this.insert(node);
          pred_group_it->descendants.erase(node);
          logical2group_it->at(node) = pred_group_it;
        }

        group_list->erase(succ_group_it);
        return true;
      }
    }
  }
  return false;
}

void ReduceGraph::BuildGraph(const LogicalGraph& logical_graph, std::list<Group>* group_list) {
  HashMap<const LogicalNode*, ReduceNode*> logical_node2reduce_node;

  for (const Group& group : *group_list) {
    ReduceNode* reduce_node = NewNode();
    reduce_node->mut_logical_nodes() = group.nodes;
    for (const LogicalNode* node : group.nodes) {
      CHECK(logical_node2reduce_node.emplace(node, reduce_node).second);
    }
  }

  std::unordered_set<std::pair<ReduceNode*, ReduceNode*>> pred_succ_pairs;

  logical_graph.ForEachEdge([&](const LogicalEdge* edge) {
    pred_succ_pairs.emplace(logical_node2reduce_node.at(edge->src_node()),
                            logical_node2reduce_node.at(edge->dst_node()));
  });

  for (auto& pair : pred_succ_pairs) {
    if (pair.first == pair.second) { continue; }
    ReduceEdge* edge = NewEdge();
    Connect(pair.first, edge, pair.second);
  }
}

bool ReduceGraph::IsLogicalNodeMergeable(const LogicalNode* logical_node) const {
  if (logical_node->parallel_desc()->policy() != kDataParallel) { return false; }

  if (!dynamic_cast<const NormalForwardLogicalNode*>(logical_node)) { return false; }

  for (const std::shared_ptr<Operator>& op : logical_node->op_vec()) {
    if (dynamic_cast<FullyConnectedOp*>(op.get())) { return false; }
    if (op->IsRecurrentOp()) { return false; }
  }

  return true;
}

}  // namespace oneflow
