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

void SetChainNodeWithChainIt(ChainNode* chain_node, ChainIt chain_it) {}

void InitChains(std::list<Chain>* chain_list,
                Logical2ChainItMap* logical2chain_it) {
  chain_list->clear();
  logical2chain_it->clear();
  LogicalGraph::Singleton()->ForEachNode([&](const LogicalNode* node) {
    // Init one Chain with one Node
    chain_list->emplace_back();
    logical2chain_it->insert({node, --chain_list->end()});
    Chain& cur_chain = chain_list->back();
    cur_chain.nodes = {node};
  });
  // Init ancestors
  LogicalGraph::Singleton()->TopoForEachNode([&](LogicalNode* node) {
    ChainIt cur_chain = logical2chain_it->at(node);
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
  LogicalGraph::Singleton()->ReverseTopoForEachNode([&](LogicalNode* node) {
    ChainIt cur_chain = logical2chain_it->at(node);
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

void ModelMergeChains(std::list<Chain>* chain_list,
                      Logical2ChainItMap* logical2chain_it) {
  for (auto& pair : *logical2chain_it) {
    // Get cur_node, pred_node
    const LogicalNode* cur_node = pair.first;
    if (cur_node->op()->IsElemWiseOp() == false) { continue; }
    if (cur_node->parallel_desc()->policy() != kModelParallel) { continue; }
    const LogicalNode* pred_node = cur_node->SoleInEdge()->src_node();
    CHECK(pred_node->parallel_desc()->Equal(cur_node->parallel_desc().get()));
    // Get chain
    ChainIt pred_chain = logical2chain_it->at(pred_node);
    ChainIt cur_chain = pair.second;
    // Merge
    pred_chain->nodes.insert(pred_chain->nodes.end(), cur_chain->nodes.begin(),
                             cur_chain->nodes.end());
    for (const LogicalNode* node : cur_chain->nodes) {
      pred_chain->descendants.erase(node);
      logical2chain_it->at(node) = pred_chain;
    }
    chain_list->erase(cur_chain);
  }
}

bool TryMergeWithConnect(const LogicalNode* up_node,
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

bool TryMergeWithoutConnect(const LogicalNode* lhs_node,
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

bool TryDataMerge(const LogicalNode* first, const LogicalNode* second,
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

bool DoOneDataMerge(const std::vector<const LogicalNode*>& data_parallel_node,
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

void DataMergeChains(std::list<Chain>* chain_list,
                     Logical2ChainItMap* logical2chain_it) {
  std::vector<const LogicalNode*> data_parallel_node;
  for (const auto& pair : *logical2chain_it) {
    const LogicalNode* cur_logi_node = pair.first;
    if (cur_logi_node->parallel_desc()->policy() != kDataParallel) { continue; }
    if (cur_logi_node->op()->IsLossOp()) { continue; }
    if (cur_logi_node->op()->IsDataLoaderOp()) { continue; }
    data_parallel_node.push_back(cur_logi_node);
  }
  while (DoOneDataMerge(data_parallel_node, chain_list, logical2chain_it)) {}
}

}  // namespace

ChainGraph::ChainGraph(bool is_train) {
  BuildFwStruct();
  if (is_train) {
    BuildBwStruct();
    BuildLossRecordStruct();
  }
  BuildModelStruct(is_train);
  BuildRnnStruct();
  ToDotWithAutoFilePath();
}

void ChainGraph::BuildFwStruct() {
  // Build Chain
  std::list<Chain> chain_list;
  Logical2ChainItMap logical2chain_it;
  InitChains(&chain_list, &logical2chain_it);
  ModelMergeChains(&chain_list, &logical2chain_it);
  DataMergeChains(&chain_list, &logical2chain_it);
  // Init chain_nodes
  auto HashChainIt = [](const ChainIt& chain_it) {
    return std::hash<Chain*>()(&(*chain_it));
  };
  HashMap<ChainIt, ChainNode*, decltype(HashChainIt)> chain_it2chain_node(
      11, HashChainIt);
  HashMap<ChainNode*, std::unordered_set<ChainNode*>> chain_node2pred;
  FOR_EACH(chain_it, chain_list) {
    ChainNode* chain_node = nullptr;
    if (chain_it->nodes.size() == 1) {
      std::shared_ptr<const Operator> op = chain_it->nodes[0]->op();
      if (op->IsLossOp()) {
        chain_node = NewNode<LossChainNode>();
      } else if (op->IsDataLoaderOp()) {
        chain_node = NewNode<SourceChainNode>();
      } else {
        // do nothing
      }
    }
    if (chain_node == nullptr) { chain_node = NewNode<ForwardChainNode>(); }
    chain_it2chain_node[chain_it] = chain_node;
    chain_node2pred[chain_node] = {};
    CHECK(!chain_it->nodes.empty());
    chain_node->mut_parallel_desc() = chain_it->nodes.front()->parallel_desc();
    for (const LogicalNode* logical_node : chain_it->nodes) {
      chain_node->mut_op_vec().push_back(logical_node->op());
    }
  }
  // Record the predecessor
  FOR_EACH(chain_it, chain_list) {
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
}

void ChainGraph::BuildBwStruct() {
  HashSet<ForwardChainNode*> fw_nodes_that_need_bw;
  TopoForEachNode([&](ChainNode* chain_node) {
    auto fw_chain_node = dynamic_cast<ForwardChainNode*>(chain_node);
    if (fw_chain_node == nullptr) { return; }
    if (fw_chain_node->HasOpWithModelOrModelTmpBlob()) {
      CHECK(fw_nodes_that_need_bw.insert(fw_chain_node).second);
      return;
    }
    for (ChainEdge* edge : fw_chain_node->in_edges()) {
      auto fw_pred_node = static_cast<ForwardChainNode*>(edge->src_node());
      if (fw_nodes_that_need_bw.find(fw_pred_node)
          != fw_nodes_that_need_bw.end()) {
        CHECK(fw_nodes_that_need_bw.insert(fw_chain_node).second);
        return;
      }
    }
  });
  for (ForwardChainNode* fw_node : fw_nodes_that_need_bw) {
    BackwardChainNode* bw_node = NewNode<BackwardChainNode>();
    bw_node->mut_op_vec() = fw_node->op_vec();
    bw_node->mut_parallel_desc() = fw_node->parallel_desc();
    fw_node->set_bw_node(bw_node);
    bw_node->set_fw_node(fw_node);
  }
  std::list<ChainEdge*> fw_edges;
  ForEachEdge([&](ChainEdge* edge) { fw_edges.push_back(edge); });
  for (ChainEdge* fw_edge : fw_edges) {
    auto fw_src_node = dynamic_cast<ForwardChainNode*>(fw_edge->src_node());
    if (fw_src_node == nullptr) { continue; }
    auto fw_dst_node = dynamic_cast<ForwardChainNode*>(fw_edge->dst_node());
    ChainNode* bw_src_node = fw_src_node->bw_node();
    if (bw_src_node == nullptr) { continue; }
    if (fw_dst_node == nullptr) {
      Connect(fw_edge->dst_node(), NewEdge(), bw_src_node);
    } else {
      ChainNode* bw_dst_node = fw_dst_node->bw_node();
      if (bw_dst_node == nullptr) { continue; }
      Connect(bw_dst_node, NewEdge(), bw_src_node);
    }
  }
  for (ForwardChainNode* fw_node : fw_nodes_that_need_bw) {
    BackwardChainNode* bw_node = fw_node->bw_node();
    Connect<ChainNode>(fw_node, NewEdge(), bw_node);
  }
}

void ChainGraph::BuildLossRecordStruct() {
  ForEachChainNode<LossChainNode>([&](LossChainNode* loss_chain) {
    // Loss Accumulate Chain
    OperatorConf loss_acc_op_conf;
    loss_acc_op_conf.set_name("loss_acc_" + NewUniqueId());
    loss_acc_op_conf.mutable_accumulate_conf();
    auto loss_acc_op = OpMgr::Singleton()->AddOp(loss_acc_op_conf);
    auto loss_acc_chain = NewNode<LossAccChainNode>();
    loss_acc_chain->mut_op_vec() = {loss_acc_op};
    loss_acc_chain->mut_parallel_desc() = loss_chain->parallel_desc();
    Connect<ChainNode>(loss_chain, NewEdge(), loss_acc_chain);
    // Loss Record Chain
    OperatorConf loss_record_op_conf;
    loss_record_op_conf.set_name("loss_record_" + NewUniqueId());
    loss_record_op_conf.mutable_loss_record_conf();
    auto loss_record_op = OpMgr::Singleton()->AddOp(loss_record_op_conf);
    ParallelConf loss_record_pr_conf;
    loss_record_pr_conf.set_policy(kDataParallel);
    loss_record_pr_conf.add_device_name(
        IDMgr::Singleton()->MachineName4MachineId(0) + ":0");
    auto loss_record_chain = NewNode<LossRecordChainNode>();
    loss_record_chain->mut_op_vec() = {loss_record_op};
    loss_record_chain->mut_parallel_desc().reset(
        new ParallelDesc(loss_record_pr_conf));
    Connect<ChainNode>(loss_acc_chain, NewEdge(), loss_record_chain);
  });
}

void ChainGraph::BuildModelStruct(bool is_train) {
  ForEachChainNode<ForwardChainNode>([&](ForwardChainNode* fw_chain) {
    if (fw_chain->HasOpWithModelOrModelTmpBlob() == false) { return; }
    // Model Update Chain
    auto md_updt_chain = NewNode<MdUpdtChainNode>();
    md_updt_chain->mut_op_vec() = {OpMgr::Singleton()->ModelUpdateOp()};
    md_updt_chain->mut_parallel_desc() = fw_chain->parallel_desc();
    Connect<ChainNode>(md_updt_chain, NewEdge(), fw_chain);
    // Model Save Chain
    OperatorConf model_save_op_conf;
    model_save_op_conf.set_name("md_save_" + NewUniqueId());
    for (std::shared_ptr<const Operator> op : fw_chain->op_vec()) {
      for (const std::string& mbn : op->model_bns()) {
        const std::string& lbn = op->Lbn4BnInOp(mbn);
        model_save_op_conf.mutable_model_save_conf()->add_lbns(lbn);
      }
    }
    auto model_save_op = OpMgr::Singleton()->AddOp(model_save_op_conf);
    auto md_save_chain = NewNode<MdSaveChainNode>();
    md_save_chain->mut_op_vec() = {model_save_op};
    auto md_save_pr_desc = new ParallelDesc(*(fw_chain->parallel_desc()));
    if (fw_chain->parallel_desc()->policy() == ParallelPolicy::kDataParallel) {
      md_save_pr_desc->RemoveNeedlessDevice(1);
    }
    md_save_chain->mut_parallel_desc().reset(md_save_pr_desc);
    Connect<ChainNode>(md_updt_chain, NewEdge(), md_save_chain);
    // Model Diff Accumulate Chain
    if (is_train == false) { return; }
    BackwardChainNode* bw_chain = fw_chain->bw_node();
    Connect<ChainNode>(md_updt_chain, NewEdge(), bw_chain);
    OperatorConf md_diff_acc_op_conf;
    md_diff_acc_op_conf.set_name("md_diff_acc_" + NewUniqueId());
    md_diff_acc_op_conf.mutable_accumulate_conf();
    auto md_diff_acc_op = OpMgr::Singleton()->AddOp(md_diff_acc_op_conf);
    auto md_diff_acc_chain = NewNode<MdDiffAccChainNode>();
    md_diff_acc_chain->mut_op_vec() = {md_diff_acc_op};
    md_diff_acc_chain->mut_parallel_desc() = fw_chain->parallel_desc();
    Connect<ChainNode>(bw_chain, NewEdge(), md_diff_acc_chain);
    Connect<ChainNode>(md_diff_acc_chain, NewEdge(), md_updt_chain);
  });
}

void ChainGraph::BuildRnnStruct() {}

}  // namespace oneflow
