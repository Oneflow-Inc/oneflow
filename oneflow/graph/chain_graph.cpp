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
  for (const std::unique_ptr<LogicalNode>& node : logi_gph.nodes()) {
    // Init one Chain with one Node
    chain_list->emplace_back();
    logical2chain_it->insert({node.get(), --chain_list->end()});
    Chain& cur_chain = chain_list->back();
    cur_chain.nodes = {node.get()};
  }
  // Init ancestors
  for (auto node = logi_gph.cbegin(); node != logi_gph.cend(); ++node) {
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
  for (auto node = logi_gph.crbegin(); node != logi_gph.crend(); ++node) {
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
    if (cur_node->op()->IsElemWise() == false) { continue; }
    if (cur_node->parallel_desc()->policy() != kModelParallel) { continue; }
    const LogicalNode* pred_node = cur_node->SoleInEdge()->src_node();
    CHECK(pred_node->parallel_desc() == cur_node->parallel_desc());
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

void Traverse(const LogicalNode* seed_node,
              const std::vector<const LogicalNode*>& data_parallel_node,
              std::list<Chain>* chain_list,
              HashMap<const LogicalNode*, bool>* done,
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
    const LogicalGraph& logical_gph,
    std::list<Chain>* chain_list,
    Logical2ChainItMap* logical2chain_it) {
  std::vector<const LogicalNode*> data_parallel_node;
  HashMap<const LogicalNode*, bool> done;
  for (const auto& pair : *logical2chain_it) {
    const LogicalNode* cur_logi_node = pair.first;
    if (cur_logi_node->parallel_desc()->policy() != kDataParallel) { continue; }
    if (logical_gph.IsFirstNode(cur_logi_node)) { continue; }
    if (cur_logi_node->IsLossNode()) { continue; }
    data_parallel_node.push_back(cur_logi_node);
    done[cur_logi_node] = false;
  }
  for (const LogicalNode* seed_node : data_parallel_node) {
    if (done.at(seed_node)) { continue; }
    Traverse(seed_node,
             data_parallel_node,
             chain_list,
             &done,
             logical2chain_it);
  }
}

} // namespace

std::string ChainNode::ConcatedOpsName() const {
  std::stringstream ss;
  ss << "ConcatedOpsName";
  for (auto op : op_vec_) {
    ss << "_" << op->op_name();
  }
  return ss.str();
}

ChainGraph::ChainGraph(const LogicalGraph* logical_gph) {
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
      chain_it2chain_node(0, HashChainIt);
  HashMap<ChainNode*, std::unordered_set<ChainNode*>> chain_node2pred;
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
      Connect(pred_node, NewFinalEdge(), cur_node);
    }
  }
  // Post processing
  UpdateSourceAndSink();
  SetInOutLbn4AllChainNodeInDataTaskGraph();
}

void ChainGraph::SetInOutLbn4AllChainNodeInDataTaskGraph() {

  // get all input_lbns_ of every chain node
  for (ChainNode& cur_node : (*this)) {

    // get all ops_output_lbns in one chain node
    std::unordered_set<std::string> ops_output_lbns;
    for (auto& op : cur_node.op_vec()) {
      for (const std::string& bn : op->output_bns()){
        ops_output_lbns.insert(op->obn2lbn(bn));
      }
    }

    // for each op_input_lbn, 
    //   if not exist in the ops_output_lbns
    //   then the op_input_lbn is the input_lbn of the chain node
    std::unordered_set<std::string> chain_node_input_lbns;
    for (auto& op : cur_node.op_vec()) {
      for (const std::string& bn : op->input_bns()) {
        std::string op_input_lbn = op->ibn2lbn(bn);
        if (ops_output_lbns.count(op_input_lbn) == 0) {
          chain_node_input_lbns.insert(op_input_lbn);
        }
      }
    }
    std::copy(chain_node_input_lbns.begin(), 
              chain_node_input_lbns.end(), 
              std::back_inserter(cur_node.mut_input_lbns()));
  }

  // get all output_lbns_ of every chain node
  // the output_lbns_ of one chain node is the sum input_lbns of it's child node
  for (ChainNode& cur_node : (*this)) {  
    std::unordered_set<std::string> chain_node_output_lbns;
    for (ChainEdge* out_edge : cur_node.out_edges()) {
      ChainNode* child_node = (out_edge)->dst_node();
      chain_node_output_lbns.insert(child_node->input_lbns().begin(), 
                                    child_node->input_lbns().end());
    }
    std::copy(chain_node_output_lbns.begin(), 
              chain_node_output_lbns.end(), 
              std::back_inserter(cur_node.mut_output_lbns()));
  }
}

std::vector<std::string> FindLbnsBetween(const ChainNode* father_node, 
                                         const ChainNode* child_node) {
  std::vector<std::string> matching_lbns;
  for (const std::string& father_node_output_lbn : father_node->output_lbns()) {
    for (const std::string& child_node_input_lbn : child_node->input_lbns()) { 
      if (father_node_output_lbn != child_node_input_lbn) {
        continue;
      }        
      matching_lbns.push_back(father_node_output_lbn);
      break;
    }
  }
  CHECK_NE(matching_lbns.size(), 0);
  return matching_lbns;
}


} // namespace oneflow
