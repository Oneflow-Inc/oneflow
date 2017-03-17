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
using Logical2ChainItMap = std::unordered_map<const LogicalNode*, ChainIt>;

void SetChainNodeWithChainIt(ChainNode* chain_node,
                             ChainIt chain_it) {
  CHECK_EQ(chain_it->nodes.empty(), false);
  chain_node->mutable_parallel_desc_ptr() =
      chain_it->nodes.front()->parallel_desc_ptr();
  for (const LogicalNode* logical_node : chain_it->nodes) {
    chain_node->mutable_op_vec().push_back(logical_node->op_ptr());
  }
}

void InitChains(
    const LogicalGraph& logi_graph,
    std::list<Chain>* chain_list,
    Logical2ChainItMap* logical2chain_it) {
  // Init one Chain with one Node
  chain_list->clear();
  logical2chain_it->clear();
  // Init ops
  for (const std::unique_ptr<LogicalNode>& node : logi_graph.nodes()) {
    chain_list->emplace_back();
    logical2chain_it->insert({node.get(), --chain_list->end()});
    Chain& cur_chainment = chain_list->back();
    cur_chainment.nodes = {node.get()};
  }
  // Init ancestors
  for (auto node = logi_graph.cbegin(); node != logi_graph.cend(); ++node) {
    ChainIt cur_chain = logical2chain_it->at(&(*node));
    cur_chain->ancestors.clear();
    // each predecessor
    for (const LogicalEdge* edge : node->in_edges()) {
      LogicalNode* pred_node = edge->src_node();
      ChainIt pred_chain = logical2chain_it->at(pred_node);
      // ancestors
      cur_chain->ancestors.insert(pred_chain->ancestors.begin(),
                                  pred_chain->ancestors.end());
      cur_chain->ancestors.insert(pred_node);
      // ancestors_and_this
      cur_chain->ancestors_and_this = cur_chain->ancestors;
      cur_chain->ancestors_and_this.insert(cur_chain->nodes.begin(),
                                           cur_chain->nodes.end());
    }
  }
  // Init descendants
  for (auto node = logi_graph.crbegin(); node != logi_graph.crend(); ++node) {
    ChainIt cur_chain = logical2chain_it->at(&(*node));
    cur_chain->descendants.clear();
    // each successors
    for (const LogicalEdge* edge : node->out_edges()) {
      LogicalNode* succ_node = edge->dst_node();
      ChainIt succ_chain = logical2chain_it->at(succ_node);
      // descendants
      cur_chain->descendants.insert(succ_chain->descendants.begin(),
                                    succ_chain->descendants.end());
      cur_chain->descendants.insert(succ_node);
      // descendants_and_this
      cur_chain->descendants_and_this = cur_chain->descendants;
      cur_chain->descendants_and_this.insert(cur_chain->nodes.begin(),
                                             cur_chain->nodes.end());
    }
  }
}

void ModelMergeChains(
    std::list<Chain>* chain_list,
    Logical2ChainItMap* logical2chain_it) {
  for (auto& pair : *logical2chain_it) {
    // Get cur_node, pred_node
    const LogicalNode* cur_node = pair.first;
    if (cur_node->op().IsElemWise() == false) {
      continue;
    }
    if (cur_node->parallel_desc().policy() != ParallelDesc::kModelParallel) {
      continue;
    }
    CHECK_EQ(cur_node->in_edges().size(), 1);
    const LogicalNode* pred_node = (*(cur_node->in_edges().begin()))->src_node();
    if (pred_node->parallel_desc() != cur_node->parallel_desc()) {
      continue;
    }
    // Get chain
    ChainIt pred_chain = logical2chain_it->at(pred_node);
    ChainIt cur_chain = pair.second;
    // Merge
    pred_chain->nodes.insert(pred_chain->nodes.end(),
                             cur_chain->nodes.begin(),
                             cur_chain->nodes.end());
    for (const LogicalNode* node : cur_chain->nodes) {
      pred_chain->descendants.erase(node);
      logical2chain_it->at(node) = pred_chain;
    }
    chain_list->erase(cur_chain);
  }
}

bool TryMergeWithConnect(
    const LogicalNode* up_node,
    const LogicalNode* bottom_node,
    std::list<Chain>* chain_list,
    Logical2ChainItMap* logical2chain_it) {
  // Get chain
  ChainIt up_chain = logical2chain_it->at(up_node);
  ChainIt bottom_chain = logical2chain_it->at(bottom_node);
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
      logical2chain_it->at(node) = up_chain;
    }
    chain_list->erase(bottom_chain);
  } else {
    for (const LogicalNode* node : up_chain->nodes) {
      bottom_chain->nodes.push_back(node);
      bottom_chain->ancestors.erase(node);
      logical2chain_it->at(node) = bottom_chain;
    }
    chain_list->erase(up_chain);
  }
  return true;
}

bool TryMergeWithoutConnect(
    const LogicalNode* lhs_node,
    const LogicalNode* rhs_node,
    std::list<Chain>* chain_list,
    Logical2ChainItMap* logical2chain_it) {
  // Get chain
  ChainIt lhs_chain = logical2chain_it->at(lhs_node);
  ChainIt rhs_chain = logical2chain_it->at(rhs_node);
  // if it can be merged
  if (lhs_chain->ancestors != rhs_chain->ancestors
      || lhs_chain->descendants != rhs_chain->descendants) {
    return false;
  }
  // Merge
  // TODO:
  // If this is bottleneck, we can optimze it by compare the size of lhs,rhs
  for (const LogicalNode* node : rhs_chain->nodes) {
    lhs_chain->nodes.push_back(node);
    lhs_chain->ancestors_and_this.insert(node);
    lhs_chain->descendants_and_this.insert(node);
    logical2chain_it->at(node) = lhs_chain;
  }
  chain_list->erase(rhs_chain);
  return true;
}

void Traverse(const LogicalNode* seed_node,
              const std::vector<const LogicalNode*>& data_parallel_node,
              std::list<Chain>* chain_list,
              std::unordered_map<const LogicalNode*, bool>* done,
              Logical2ChainItMap* logical2chain_it) {
  done->at(seed_node) = true;
  while (true) {
    bool has_merged = false;
    for (const LogicalNode* node : data_parallel_node) {
      if (done->at(node)) { continue; }
      if (seed_node->parallel_desc() != node->parallel_desc()) {
        continue;
      }
      if (TryMergeWithConnect(seed_node, node, chain_list, logical2chain_it)
          || TryMergeWithConnect(node, seed_node, chain_list, logical2chain_it)
          || TryMergeWithoutConnect(seed_node, node, chain_list, logical2chain_it)) {
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
    const LogicalGraph& logical_graph,
    std::list<Chain>* chain_list,
    Logical2ChainItMap* logical2chain_it) {
  std::vector<const LogicalNode*> data_parallel_node;
  std::unordered_map<const LogicalNode*, bool> done;
  for (const auto& pair : *logical2chain_it) {
    if (pair.first->parallel_desc().policy() == ParallelDesc::kDataParallel
        && !logical_graph.IsFirstNode(pair.first)) {
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
               logical2chain_it);
    }
  }
}

} // namespace

void ChainGraph::Init(std::shared_ptr<const LogicalGraph> logical_graph) {
  // Build Chain
  std::list<Chain> chain_list;
  Logical2ChainItMap logical2chain_it;
  InitChains(*logical_graph, &chain_list, &logical2chain_it);
  ModelMergeChains(&chain_list, &logical2chain_it);
  DataMergeChains(*logical_graph,
                  &chain_list,
                  &logical2chain_it);
  // Init chain_nodes
  auto hash_chain_it = [](const ChainIt& chain_it) {
    return std::hash<Chain*> ()(&(*chain_it));
  };
  std::unordered_map<ChainIt, ChainNode*, decltype(hash_chain_it)>
      chain_it2chain_node(0, hash_chain_it);
  std::unordered_map<ChainNode*,
                     std::unordered_set<ChainNode*>> chain_node2pred;
  for (auto chain_it = chain_list.begin(); chain_it != chain_list.end(); ++chain_it) {
    ChainNode* chain_node = NewFinalNode();
    chain_it2chain_node[chain_it] = chain_node;
    chain_node2pred[chain_node] = {};
    SetChainNodeWithChainIt(chain_node, chain_it);
  }
  // Record the predecessor
  for (auto chain_it = chain_list.begin(); chain_it != chain_list.end(); ++chain_it) {
    ChainNode* chain_node = chain_it2chain_node.at(chain_it);
    for (const LogicalNode* logi_node : chain_it->nodes) {
      for (auto logi_in_edge : logi_node->in_edges()) {
        auto pred_chain_it = logical2chain_it.at(logi_in_edge->src_node());
        auto pred_chain_node = chain_it2chain_node.at(pred_chain_it);
        chain_node2pred.at(chain_node).insert(pred_chain_node);
      }
    }
  }
  // Connect
  for (auto& pair : chain_node2pred) {
    ChainNode* cur_node = pair.first;
    for (ChainNode* pred_node : pair.second) {
      Connect(pred_node, NewFinalEdge(), cur_node);
    }
  }
  // Post processing
  UpdateStartAndStop();
  CollectInputAndOutputLbns();
}

void ChainGraph::CollectInputAndOutputLbns() {
  // set input_lbns_ and output_lbns_ for each node
  LOG(FATAL) << "TODO";
}

} // namespace oneflow
