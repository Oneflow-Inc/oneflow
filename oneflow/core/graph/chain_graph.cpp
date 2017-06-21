#include "oneflow/core/graph/chain_graph.h"

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
using Logical2ChainItMap = HashMap<const LogicalNode*, ChainIt>;

void SetChainNodeWithChainIt(ChainNode* chain_node, ChainIt chain_it) {
  CHECK(!chain_it->nodes.empty());
  chain_node->mut_parallel_desc() = chain_it->nodes.front()->parallel_desc();
  for (const LogicalNode* logical_node : chain_it->nodes) {
    chain_node->mut_op_vec().push_back(logical_node->op());
  }
}

void InitChains(
    const LogicalGraph& logi_gph,
    std::list<Chain>* chain_list,
    Logical2ChainItMap* logical2chain_it) {
  chain_list->clear();
  logical2chain_it->clear();
  logi_gph.ConstForEachNode([&](const LogicalNode* node) {
    // Init one Chain with one Node
    chain_list->emplace_back();
    logical2chain_it->insert({node, --chain_list->end()});
    Chain& cur_chain = chain_list->back();
    cur_chain.nodes = {node};
  });
  // Init ancestors
  logi_gph.ConstTopoForEachNode([&](const LogicalNode* node) {
    ChainIt cur_chain = logical2chain_it->at(&(*node));
    cur_chain->ancestors.clear();
    cur_chain->ancestors_and_this.clear();
    cur_chain->ancestors_and_this.insert(cur_chain->nodes.begin(),
                                         cur_chain->nodes.end());
    // each predecessor
    for (const LogicalEdge* edge : node->in_edges()) {
      LogicalNode* pred_node = edge->src_node();
      ChainIt pred_chain = logical2chain_it->at(pred_node);
      // ancestors
      cur_chain->ancestors.insert(pred_chain->ancestors.begin(),
                                  pred_chain->ancestors.end());
      cur_chain->ancestors.insert(pred_node);
    }
    // ancestors_and_this
    cur_chain->ancestors_and_this.insert(cur_chain->ancestors.begin(),
                                         cur_chain->ancestors.end());
  });
  // Init descendants
  logi_gph.ConstReverseTopoForEachNode([&](const LogicalNode* node) {
    ChainIt cur_chain = logical2chain_it->at(&(*node));
    cur_chain->descendants.clear();
    cur_chain->descendants_and_this.clear();
    cur_chain->descendants_and_this.insert(cur_chain->nodes.begin(),
                                           cur_chain->nodes.end());
    // each successors
    for (const LogicalEdge* edge : node->out_edges()) {
      LogicalNode* succ_node = edge->dst_node();
      ChainIt succ_chain = logical2chain_it->at(succ_node);
      // descendants
      cur_chain->descendants.insert(succ_chain->descendants.begin(),
                                    succ_chain->descendants.end());
      cur_chain->descendants.insert(succ_node);
      // descendants_and_this
    }
    cur_chain->descendants_and_this.insert(cur_chain->descendants.begin(),
                                           cur_chain->descendants.end());
  });
}

void ModelMergeChains(
    std::list<Chain>* chain_list,
    Logical2ChainItMap* logical2chain_it) {
  for (auto& pair : *logical2chain_it) {
    // Get cur_node, pred_node
    const LogicalNode* cur_node = pair.first;
    if (cur_node->op()->IsElemWise() == false) { continue; }
    if (cur_node->parallel_desc()->policy() != kModelParallel) { continue; }
    const LogicalNode* pred_node = cur_node->SoleInEdge()->src_node();
    CHECK(pred_node->parallel_desc()->Equal(cur_node->parallel_desc().get()))
        << "the ParallelConf of "
        << "\"" << pred_node->op()->op_name() << "\" "
        << "and "
        << "\"" << cur_node->op()->op_name() << "\" "
        << "should be the same";
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
      up_chain->ancestors_and_this.insert(node);
      up_chain->descendants.erase(node);
      logical2chain_it->at(node) = up_chain;
    }
    chain_list->erase(bottom_chain);
  } else {
    for (const LogicalNode* node : up_chain->nodes) {
      bottom_chain->nodes.push_back(node);
      bottom_chain->descendants_and_this.insert(node);
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
  // if it cannot be merged
  if (lhs_chain->ancestors != rhs_chain->ancestors
      || lhs_chain->descendants != rhs_chain->descendants) {
    return false;
  }
  // Merge
  for (const LogicalNode* node : rhs_chain->nodes) {
    lhs_chain->nodes.push_back(node);
    lhs_chain->ancestors_and_this.insert(node);
    lhs_chain->descendants_and_this.insert(node);
    logical2chain_it->at(node) = lhs_chain;
  }
  chain_list->erase(rhs_chain);
  return true;
}

bool TryDataMerge(
    const LogicalNode* first,
    const LogicalNode* second,
    std::list<Chain>* chain_list,
    Logical2ChainItMap* logical2chain_it) {
  if (first->parallel_desc()->Equal(second->parallel_desc().get()) == false) {
    return false;
  }
  if (TryMergeWithoutConnect(first, second, chain_list, logical2chain_it)
      || TryMergeWithConnect(first, second, chain_list, logical2chain_it)
      || TryMergeWithConnect(second, first, chain_list, logical2chain_it)) {
    return true;
  }
  return false;
}

bool DoOneDataMerge(
    const std::vector<const LogicalNode*>& data_parallel_node,
    std::list<Chain>* chain_list,
    Logical2ChainItMap* logical2chain_it) {
  for (const LogicalNode* first : data_parallel_node) {
    for (const LogicalNode* second : data_parallel_node) {
      if (first == second) { continue; }
      ChainIt first_it = logical2chain_it->at(first);
      ChainIt second_it = logical2chain_it->at(second);
      if (first_it == second_it) { continue; }
      if (TryDataMerge(first, second, chain_list, logical2chain_it)) {
        return true;
      }
    }
  }
  return false;
}

void DataMergeChains(
    const LogicalGraph& logical_gph,
    std::list<Chain>* chain_list,
    Logical2ChainItMap* logical2chain_it) {
  std::vector<const LogicalNode*> data_parallel_node;
  for (const auto& pair : *logical2chain_it) {
    const LogicalNode* cur_logi_node = pair.first;
    if (cur_logi_node->parallel_desc()->policy() != kDataParallel) { continue; }
    if (cur_logi_node->IsLossNode()) { continue; }
    data_parallel_node.push_back(cur_logi_node);
  }
  while (DoOneDataMerge(data_parallel_node, chain_list, logical2chain_it)) {
  }
}

} // namespace

std::string ChainNode::ConcatedOpsName() const {
  std::stringstream ss;
  for (auto op : op_vec_) {
    ss << "\\n" << op->op_name();
  }
  if (!op_vec_.empty()) {
    return ss.str().substr(2);
  } else {
    return node_id_str();
  }
}

bool ChainNode::HasOpWithModelOrModelTmpBlob() const {
  for (std::shared_ptr<const Operator> op : op_vec_) {
    if (!op->model_bns().empty() || !op->model_tmp_bns().empty()) {
      return true;
    }
  }
  return false;
}

ChainGraph::ChainGraph(const LogicalGraph* logical_gph,
                       const std::string& dot_filepath) {
  LOG(INFO) << "Build ChainGraph...";
  // Build Chain
  std::list<Chain> chain_list;
  Logical2ChainItMap logical2chain_it;
  InitChains(*logical_gph, &chain_list, &logical2chain_it);
  ModelMergeChains(&chain_list, &logical2chain_it);
  DataMergeChains(*logical_gph, &chain_list, &logical2chain_it);
  // Init chain_nodes
  auto HashChainIt = [](const ChainIt& chain_it) {
    return std::hash<Chain*> ()(&(*chain_it));
  };
  HashMap<ChainIt, ChainNode*, decltype(HashChainIt)>
      chain_it2chain_node(11, HashChainIt);
  HashMap<ChainNode*, std::unordered_set<ChainNode*>> chain_node2pred;
  for (auto chain_it = chain_list.begin(); chain_it != chain_list.end(); ++chain_it) {
    ChainNode* chain_node = NewNode();
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
        if (pred_chain_node != chain_node) {
          chain_node2pred.at(chain_node).insert(pred_chain_node);
        }
      }
    }
  }
  // Connect
  for (auto& pair : chain_node2pred) {
    ChainNode* cur_node = pair.first;
    for (ChainNode* pred_node : pair.second) {
      Connect(pred_node, NewEdge(), cur_node);
    }
  }
  // Post processing
  UpdateSourceAndSink();
  SetInOutLbn4AllChainNodeInDataTaskGraph();
  ToDotFile(dot_filepath);
}

void ChainGraph::SetInOutLbn4AllChainNodeInDataTaskGraph() {
  HashMap<ChainNode*, std::unordered_set<std::string>> chain2produced_lbns;
  // Init chain2produced_lbns and Set InputLbns
  ForEachNode([&](ChainNode* cur_node) {
    auto& produced_lbns = chain2produced_lbns[cur_node];
    for (std::shared_ptr<const Operator> op : cur_node->op_vec()) {
      for (const std::string& obn : op->output_bns()) {
        std::string lbn = op->Lbn4BnInOp(obn);
        produced_lbns.insert(lbn);
      }
    }
    for (std::shared_ptr<const Operator> op : cur_node->op_vec()) {
      for (const std::string& ibn : op->input_bns()) {
        std::string lbn = op->Lbn4BnInOp(ibn);
        if (produced_lbns.find(lbn) == produced_lbns.end()) {
          cur_node->mut_input_lbns().push_back(lbn);
        }
      }
    }
    SortAndRemoveDuplication(&(cur_node->mut_input_lbns()));
  });
  // Set OutputLbns
  ForEachNode([&](ChainNode* cur_node) {
    const auto& produced_lbns = chain2produced_lbns.at(cur_node);
    for (ChainEdge* out_edge : cur_node->out_edges()) {
      for (const std::string& lbn : out_edge->dst_node()->input_lbns()) {
        if (produced_lbns.find(lbn) != produced_lbns.end()) {
          cur_node->mut_output_lbns().push_back(lbn);
        }
      }
    }
    SortAndRemoveDuplication(&(cur_node->mut_output_lbns()));
  });
}

std::vector<std::string> FindLbnsBetween(const ChainNode* src_node, 
                                         const ChainNode* dst_node) {
  std::vector<std::string> matching_lbns;
  for (const std::string& src_node_output_lbn : src_node->output_lbns()) {
    for (const std::string& dst_node_input_lbn : dst_node->input_lbns()) { 
      if (src_node_output_lbn != dst_node_input_lbn) {
        continue;
      }        
      matching_lbns.push_back(src_node_output_lbn);
      break;
    }
  }
  CHECK_NE(matching_lbns.size(), 0);
  return matching_lbns;
}

std::string ChainEdge::VisualStr() const {
  std::vector<std::string> lbns = FindLbnsBetween(src_node(), dst_node());
  std::stringstream ss;
  for (const std::string& lbn : lbns) {
    ss << "\\n" << lbn;
  }
  return ss.str().substr(2);
}

} // namespace oneflow
