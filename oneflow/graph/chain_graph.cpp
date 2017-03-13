#include "graph/chain_graph.h"
#include "glog/logging.h"
#include <list>

namespace oneflow {

namespace {

struct Chain {
  // nodes belong to this Chain
  std::vector<const LogicalNode*> nodes;
  // ancestors, descendants of nodes
  std::unordered_set<const LogicalNode*> ancestors;
  std::unordered_set<const LogicalNode*> descendants;
  // ancestors_and_this = nodes + ancestors
  // descendants_and_this = nodes + descendants
  std::unordered_set<const LogicalNode*> ancestors_and_this;
  std::unordered_set<const LogicalNode*> descendants_and_this;
};

using ChainIt = std::list<Chain>::iterator;

void SetChainNodeWithChainIt(ChainNode* chain_node,
                             ChainIt chain_it) {
  CHECK_EQ(chain_it->nodes.empty(), false);
  chain_node->mutable_parallel_desc_ptr() =
      chain_it->nodes.front()->parallel_desc_ptr();
  chain_node->mutable_op_vec_ptr().reset(new std::vector<std::shared_ptr<const Operator>>);
  for (const LogicalNode* logical_node : chain_it->nodes) {
    chain_node->mutable_op_vec().push_back(
        logical_node->op_ptr());
  }
}

void InitChains(
    const LogicalGraph* logical_graph,
    std::list<Chain>* chain_list,
    std::unordered_map<const LogicalNode*,
                       ChainIt>* logical_node2chain_it) {
  // Init one Chain with one Node
  chain_list->clear();
  logical_node2chain_it->clear();
  // Init ops
  for (const std::unique_ptr<Node>& node : logical_graph->node_vec()) {
    auto logical_node = of_dynamic_cast<const LogicalNode*>(node.get());
    chain_list->emplace_back();
    logical_node2chain_it->insert({logical_node, --chain_list->end()});
    Chain& cur_chainment = chain_list->back();
    cur_chainment.nodes = {logical_node};
  }
  // Init ancestors
  for (auto it = logical_graph->cbegin(); it != logical_graph->cend(); ++it) {
    // Get correct ptr
    auto cur_node = of_dynamic_cast<const LogicalNode*> (&(*it));
    ChainIt cur_chain = logical_node2chain_it->at(cur_node);
    cur_chain->ancestors.clear();
    // each predecessor
    for (const Edge* edge : cur_node->in_edges()) {
      auto logi_pre = of_dynamic_cast<const LogicalNode*> (edge->src_node());
      ChainIt pre_chain = logical_node2chain_it->at(logi_pre);
      // ancestors
      cur_chain->ancestors.insert(pre_chain->ancestors.begin(),
                                    pre_chain->ancestors.end());
      cur_chain->ancestors.insert(logi_pre);
      // ancestors_and_this
      cur_chain->ancestors_and_this = cur_chain->ancestors;
      cur_chain->ancestors_and_this.insert(cur_chain->nodes.begin(),
                                             cur_chain->nodes.end());
    }
  }
  // Init descendants
  for (auto it = logical_graph->crbegin(); it != logical_graph->crend(); ++it) {
    auto cur_node = of_dynamic_cast<const LogicalNode*> (&(*it));
    ChainIt cur_chain = logical_node2chain_it->at(cur_node);
    cur_chain->descendants.clear();
    // each successors
    for (const Edge* edge : cur_node->out_edges()) {
      auto logi_succ = of_dynamic_cast<const LogicalNode*> (edge->dst_node());
      ChainIt next_chain = logical_node2chain_it->at(logi_succ);
      // descendants
      cur_chain->descendants.insert(next_chain->descendants.begin(),
                                      next_chain->descendants.end());
      cur_chain->descendants.insert(logi_succ);
      // descendants_and_this
      cur_chain->descendants_and_this = cur_chain->descendants;
      cur_chain->descendants_and_this.insert(cur_chain->nodes.begin(),
                                               cur_chain->nodes.end());
    }
  }
}

void ModelMergeChains(
    std::list<Chain>* chain_list,
    std::unordered_map<const LogicalNode*,
                       ChainIt>* logical_node2chain_it) {
  for (auto& pair : *logical_node2chain_it) {
    // Get cur_node, pre_node
    const LogicalNode* cur_node = pair.first;
    if (cur_node->op().IsElemWise() == false) {
      continue;
    }
    if (cur_node->parallel_desc().policy() != ParallelDesc::kModelParallel) {
      continue;
    }
    CHECK_EQ(cur_node->in_edges().size(), 1);
    CHECK_EQ(cur_node->in_edges().size(), 1);
    auto pre_node =
        of_dynamic_cast<const LogicalNode*>
            ((*(cur_node->in_edges().begin()))->src_node());
    if (pre_node->parallel_desc() != cur_node->parallel_desc()) {
      continue;
    }
    // Get chain
    ChainIt pre_chain = logical_node2chain_it->at(pre_node);
    ChainIt cur_chain = pair.second;
    // Merge
    pre_chain->nodes.insert(pre_chain->nodes.end(),
                                 cur_chain->nodes.begin(),
                                 cur_chain->nodes.end());
    for (const LogicalNode* node : cur_chain->nodes) {
      pre_chain->descendants.erase(node);
      logical_node2chain_it->at(node) = pre_chain;
    }
    chain_list->erase(cur_chain);
  }
}

bool TryMergeWithConnect(
    const LogicalNode* up_node,
    const LogicalNode* bottom_node,
    std::list<Chain>* chain_list,
    std::unordered_map<const LogicalNode*,
                       ChainIt>* logical_node2chain_it) {
  // Get chain
  ChainIt up_chain = logical_node2chain_it->at(up_node);
  ChainIt bottom_chain = logical_node2chain_it->at(bottom_node);
  // if it can be merged
  if (up_chain->ancestors_and_this != bottom_chain->ancestors
      || bottom_chain->descendants_and_this != up_chain->descendants) {
    return false;
  }
  // Merge
  if (up_chain->nodes.size() > bottom_chain->nodes.size()) {
    for (const LogicalNode* node : bottom_chain->nodes) {
      up_chain->nodes.push_back(node);
      up_chain->descendants.erase(node);
      logical_node2chain_it->at(node) = up_chain;
    }
    chain_list->erase(bottom_chain);
  } else {
    for (const LogicalNode* node : up_chain->nodes) {
      bottom_chain->nodes.push_back(node);
      bottom_chain->ancestors.erase(node);
      logical_node2chain_it->at(node) = bottom_chain;
    }
    chain_list->erase(up_chain);
  }
  return true;
}

bool TryMergeWithoutConnect(
    const LogicalNode* lhs_node,
    const LogicalNode* rhs_node,
    std::list<Chain>* chain_list,
    std::unordered_map<const LogicalNode*,
                       ChainIt>* logical_node2chain_it) {
  // Get chain
  ChainIt lhs_chain = logical_node2chain_it->at(lhs_node);
  ChainIt rhs_chain = logical_node2chain_it->at(rhs_node);
  // if it can be merged
  if (lhs_chain->ancestors != rhs_chain->ancestors
      || lhs_chain->descendants != rhs_chain->descendants) {
    return false;
  }
  // Merge
  // If this is bottleneck, we can optimze it by compare the size of lhs,rhs
  for (const LogicalNode* node : rhs_chain->nodes) {
    lhs_chain->nodes.push_back(node);
    lhs_chain->ancestors_and_this.insert(node);
    lhs_chain->descendants_and_this.insert(node);
    logical_node2chain_it->at(node) = lhs_chain;
  }
  chain_list->erase(rhs_chain);
  return true;
}

void Traverse(const LogicalNode* seed_node,
              const std::vector<const LogicalNode*>& data_parallel_node,
              std::list<Chain>* chain_list,
              std::unordered_map<const LogicalNode*, bool>* done,
              std::unordered_map<const LogicalNode*,
                                 ChainIt>* logical_node2chain_it) {
  done->at(seed_node) = true;
  while (true) {
    bool has_merged = false;
    for (const LogicalNode* node : data_parallel_node) {
      if (done->at(node)) { continue; }
      if (seed_node->parallel_desc() != node->parallel_desc()) {
        continue;
      }
      if (TryMergeWithConnect(seed_node, node, chain_list, logical_node2chain_it)
          || TryMergeWithConnect(node, seed_node, chain_list, logical_node2chain_it)
          || TryMergeWithoutConnect(seed_node, node, chain_list, logical_node2chain_it)) {
        done->at(node) = true;
        has_merged = true;
        break;
      }
    }
    if (has_merged == false) {
      break;
    }
  }
}

void DataMergeChains(
    const LogicalGraph* logical_graph,
    std::list<Chain>* chain_list,
    std::unordered_map<const LogicalNode*,
                       ChainIt>* logical_node2chain_it) {
  std::vector<const LogicalNode*> data_parallel_node;
  std::unordered_map<const LogicalNode*, bool> done;
  for (const auto& pair : *logical_node2chain_it) {
    if (pair.first->parallel_desc().policy() == ParallelDesc::kDataParallel
        && logical_graph->IsFirstNode(pair.first) == false) {
      data_parallel_node.push_back(pair.first);
      done[pair.first] = false;
    }
  }
  for (const LogicalNode* seed_node : data_parallel_node) {
    if (done.at(seed_node) == false) {
      Traverse(seed_node,
               data_parallel_node,
               chain_list,
               &done,
               logical_node2chain_it);
    }
  }
}

} // namespace

void ChainGraph::Init(const LogicalGraph* logical_graph) {
  // Build Chain
  std::list<Chain> chain_list;
  std::unordered_map<const LogicalNode*,
                     ChainIt> logical_node2chain_it;
  InitChains(logical_graph, &chain_list, &logical_node2chain_it);
  ModelMergeChains(&chain_list, &logical_node2chain_it);
  DataMergeChains(logical_graph,
                    &chain_list,
                    &logical_node2chain_it);
  // Init chain_nodes
  auto chain_it_hash = [](const ChainIt& chain_it) {
    return std::hash<Chain*> ()(&(*chain_it));
  };
  std::unordered_map<ChainIt, ChainNode*, decltype(chain_it_hash)>
      chain_it2chain_node(0, chain_it_hash);
  std::unordered_map<ChainNode*,
                     std::unordered_set<ChainNode*>> chain_node2pre;
  for (auto chain_it = chain_list.begin(); chain_it != chain_list.end(); ++chain_it) {
    ChainNode* chain_node = NewChainNode();
    chain_it2chain_node[chain_it] = chain_node;
    chain_node2pre[chain_node] = {};
    SetChainNodeWithChainIt(chain_node, chain_it);
  }
  // Record the predecessor
  for (auto chain_it = chain_list.begin(); chain_it != chain_list.end(); ++chain_it) {
    ChainNode* chain_node = chain_it2chain_node.at(chain_it);
    for (const LogicalNode* logi_node : chain_it->nodes) {
      for (auto logi_in_edge : logi_node->in_edges()) {
        auto pre_chain_it =
            logical_node2chain_it.at(
                of_dynamic_cast<const LogicalNode*>(logi_in_edge->src_node()));
        auto pre_chain_node = chain_it2chain_node.at(pre_chain_it);
        chain_node2pre.at(chain_node).insert(pre_chain_node);
      }
    }
  }
  // Connect
  for (auto& pair : chain_node2pre) {
    ChainNode* cur_node = pair.first;
    for (ChainNode* pre_node : pair.second) {
      Connect(pre_node, NewChainEdge(), cur_node);
    }
  }
  // Post processing
  UpdateStartAndStop();
}

} // namespace oneflow
