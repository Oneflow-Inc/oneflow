#include "oneflow/core/operator/fully_connected_op.h"
#include "oneflow/core/graph/chain_logical_graph.h"
#include "oneflow/core/graph/logical_graph.h"

namespace oneflow {

struct ChainLogicalGraph::Chain {
  std::vector<const LogicalNode*> nodes;
  HashSet<const LogicalNode*> ancestors;
  HashSet<const LogicalNode*> ancestors_and_this;
  HashSet<const LogicalNode*> descendants;
  HashSet<const LogicalNode*> descendants_and_this;
  bool is_mergeable;

  bool IsParallelDescEqual(const Chain& rhs) const {
    CHECK_GT(nodes.size(), 0);
    CHECK_GT(rhs.nodes.size(), 0);
    return nodes.front()->parallel_desc()->Equal(rhs.nodes.front()->parallel_desc().get());
  }
};

ChainLogicalGraph::ChainLogicalGraph(const LogicalGraph& logical_graph) {
  std::list<Chain> chain_list;
  HashMap<const LogicalNode*, std::list<Chain>::iterator> logical2chain_it;
  HashMap<const LogicalNode*, size_t> logical2order_in_topo;

  InitChains(logical_graph, &chain_list, &logical2chain_it, &logical2order_in_topo);
  MergeChains(&chain_list, &logical2chain_it);
  SortNodesInChains(&chain_list, logical2order_in_topo);
  BuildGraph(logical_graph, &chain_list);
  ToDotWithAutoFilePath();
}

void ChainLogicalGraph::InitChains(
    const LogicalGraph& logical_graph, std::list<Chain>* chain_list,
    HashMap<const LogicalNode*, std::list<Chain>::iterator>* logical2chain_it,
    HashMap<const LogicalNode*, size_t>* logical2order_in_topo) {
  logical_graph.ForEachNode([&](const LogicalNode* node) {
    chain_list->emplace_back();
    logical2chain_it->insert({node, --chain_list->end()});
    Chain& chain = chain_list->back();
    chain.nodes = {node};
    chain.is_mergeable = IsLogicalNodeMergeable(node);
    size_t order_in_topo = logical2order_in_topo->size();
    logical2order_in_topo->emplace(node, order_in_topo);
  });

  logical_graph.TopoForEachNode([&](const LogicalNode* node) {
    auto cur_chain = logical2chain_it->at(node);
    for (const LogicalEdge* edge : node->in_edges()) {
      LogicalNode* pred_node = edge->src_node();
      auto pred_chain = logical2chain_it->at(pred_node);
      cur_chain->ancestors.insert(pred_chain->ancestors.begin(), pred_chain->ancestors.end());
      cur_chain->ancestors.insert(pred_node);
    }
    cur_chain->ancestors_and_this.insert(cur_chain->ancestors.begin(), cur_chain->ancestors.end());
    cur_chain->ancestors_and_this.insert(cur_chain->nodes.begin(), cur_chain->nodes.end());
  });

  logical_graph.ReverseTopoForEachNode([&](const LogicalNode* node) {
    auto cur_chain = logical2chain_it->at(node);
    for (const LogicalEdge* edge : node->out_edges()) {
      LogicalNode* succ_node = edge->dst_node();
      auto succ_chain = logical2chain_it->at(succ_node);
      cur_chain->descendants.insert(succ_chain->descendants.begin(), succ_chain->descendants.end());
      cur_chain->descendants.insert(succ_node);
    }
    cur_chain->descendants_and_this.insert(cur_chain->descendants.begin(),
                                           cur_chain->descendants.end());
    cur_chain->descendants_and_this.insert(cur_chain->nodes.begin(), cur_chain->nodes.end());
  });
}

void ChainLogicalGraph::MergeChains(
    std::list<Chain>* chain_list,
    HashMap<const LogicalNode*, std::list<Chain>::iterator>* logical2chain_it) {
  while (chain_list->size() > 1 && TryMergeTwoChains(chain_list, logical2chain_it)) {};
}

bool ChainLogicalGraph::TryMergeTwoChains(
    std::list<Chain>* chain_list,
    HashMap<const LogicalNode*, std::list<Chain>::iterator>* logical2chain_it) {
  return TryMergeTwoParallelChains(chain_list, logical2chain_it)
         || TryMergeTwoConnectedChains(chain_list, logical2chain_it);
}

bool ChainLogicalGraph::TryMergeTwoParallelChains(
    std::list<Chain>* chain_list,
    HashMap<const LogicalNode*, std::list<Chain>::iterator>* logical2chain_it) {
  for (auto lhs = chain_list->begin(); lhs != chain_list->end(); ++lhs) {
    if (!lhs->is_mergeable) { continue; }
    for (auto rhs = lhs; rhs != chain_list->end(); ++rhs) {
      if (lhs == rhs) { continue; }
      if (!rhs->is_mergeable) { continue; }
      if (!lhs->IsParallelDescEqual(*rhs)) { continue; }
      if (lhs->ancestors != rhs->ancestors || lhs->descendants != rhs->descendants) { continue; }
      for (const LogicalNode* node : rhs->nodes) {
        lhs->nodes.push_back(node);
        lhs->ancestors_and_this.insert(node);
        lhs->descendants_and_this.insert(node);
        logical2chain_it->at(node) = lhs;
      }
      chain_list->erase(rhs);
      return true;
    }
  }
  return false;
}

bool ChainLogicalGraph::TryMergeTwoConnectedChains(
    std::list<Chain>* chain_list,
    HashMap<const LogicalNode*, std::list<Chain>::iterator>* logical2chain_it) {
  for (auto succ_chain_it = chain_list->begin(); succ_chain_it != chain_list->end();
       ++succ_chain_it) {
    if (!succ_chain_it->is_mergeable) { continue; }
    for (const LogicalNode* node_in_succ : succ_chain_it->nodes) {
      for (const LogicalEdge* in_edge : node_in_succ->in_edges()) {
        auto pred_chain_it = logical2chain_it->at(in_edge->src_node());
        if (pred_chain_it == succ_chain_it) { continue; }
        if (!pred_chain_it->is_mergeable) { continue; }
        if (!pred_chain_it->IsParallelDescEqual(*succ_chain_it)) { continue; }
        if (pred_chain_it->ancestors_and_this != succ_chain_it->ancestors
            || pred_chain_it->descendants != succ_chain_it->descendants_and_this) {
          continue;
        }
        for (const LogicalNode* node : succ_chain_it->nodes) {
          pred_chain_it->nodes.push_back(node);
          pred_chain_it->ancestors_and_this.insert(node);
          pred_chain_it->descendants.erase(node);
          logical2chain_it->at(node) = pred_chain_it;
        }
        chain_list->erase(succ_chain_it);
        return true;
      }
    }
  }
  return false;
}

void ChainLogicalGraph::SortNodesInChains(
    std::list<Chain>* chain_list,
    const HashMap<const LogicalNode*, size_t>& logical2order_in_topo) {
  for (Chain& chain : *chain_list) {
    std::sort(chain.nodes.begin(), chain.nodes.end(),
              [&](const LogicalNode* a, const LogicalNode* b) {
                return logical2order_in_topo.at(a) < logical2order_in_topo.at(b);
              });
  }
}

void ChainLogicalGraph::BuildGraph(const LogicalGraph& logical_graph,
                                   std::list<Chain>* chain_list) {
  HashMap<const LogicalNode*, ChainLogicalNode*> logical_node2chain_logical_node;

  for (const Chain& chain : *chain_list) {
    ChainLogicalNode* chain_logical_node = NewNode();
    chain_logical_node->mut_logical_nodes() = chain.nodes;
    for (const LogicalNode* node : chain.nodes) {
      CHECK(logical_node2chain_logical_node.emplace(node, chain_logical_node).second);
    }
  }

  std::unordered_set<std::pair<ChainLogicalNode*, ChainLogicalNode*>> pred_succ_pairs;
  logical_graph.ForEachEdge([&](const LogicalEdge* edge) {
    pred_succ_pairs.emplace(logical_node2chain_logical_node.at(edge->src_node()),
                            logical_node2chain_logical_node.at(edge->dst_node()));
  });

  for (auto& pair : pred_succ_pairs) {
    if (pair.first == pair.second) { continue; }
    ChainLogicalEdge* edge = NewEdge();
    Connect(pair.first, edge, pair.second);
  }
}

bool ChainLogicalGraph::IsLogicalNodeMergeable(const LogicalNode* logical_node) const {
  if (logical_node->parallel_desc()->policy() != kDataParallel) { return false; }
  for (const std::shared_ptr<Operator>& op : logical_node->op_vec()) {
    if (dynamic_cast<FullyConnectedOp*>(op.get())) { return true; }
    if (op->IsRecurrentOp()) { return false; }
  }
  return true;
}

}  // namespace oneflow
