#include "oneflow/core/graph/chain_graph.h"

namespace oneflow {

namespace {

struct Chain {
  // nodes belong to this Chain
  std::vector<const LogicalNode*> nodes;
  // ancestors, descendants of nodes
  HashSet<const LogicalNode*> ancestors;
  HashSet<const LogicalNode*> descendants;
  // ancestors_and_this = nodes + ancestors
  // descendants_and_this = nodes + descendants
  HashSet<const LogicalNode*> ancestors_and_this;
  HashSet<const LogicalNode*> descendants_and_this;
};

using ChainIt = std::list<Chain>::iterator;
using Logical2ChainItMap = HashMap<const LogicalNode*, ChainIt>;

void SetChainNodeWithChainIt(ChainNode* chain_node, ChainIt chain_it) {}

void ModifyOpLbn4BnInChainNode(
    const HashMap<std::string, std::string>& olbn2ilbn, ChainNode* chain_node) {
  for (std::shared_ptr<Operator> op : chain_node->op_vec()) {
    for (const std::string& ibn : op->input_bns()) {
      const std::string& lbn = op->Lbn4BnInOp(ibn);
      auto olbn2ilbn_it = olbn2ilbn.find(lbn);
      if (olbn2ilbn_it == olbn2ilbn.end()) { continue; }
      op->ModifyLbn4BnInOp(ibn, olbn2ilbn_it->second);
    }
  }
}

void InitChains(std::list<Chain>* chain_list,
                Logical2ChainItMap* logical2chain_it) {
  chain_list->clear();
  logical2chain_it->clear();
  Global<LogicalGraph>::Get()->ForEachNode([&](const LogicalNode* node) {
    // Init one Chain with one Node
    chain_list->emplace_back();
    logical2chain_it->insert({node, --chain_list->end()});
    Chain& cur_chain = chain_list->back();
    cur_chain.nodes = {node};
  });
  // Init ancestors
  Global<LogicalGraph>::Get()->TopoForEachNode([&](LogicalNode* node) {
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
  Global<LogicalGraph>::Get()->ReverseTopoForEachNode([&](LogicalNode* node) {
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
    if (!pred_node->parallel_desc()->Equal(cur_node->parallel_desc().get())) {
      continue;
    }
    if (pred_node->op()->IsRecurrentOp()) { continue; }
    if (pred_node->shared_model_nodes()) { continue; }
    // Get chain
    ChainIt pred_chain = logical2chain_it->at(pred_node);
    ChainIt cur_chain = pair.second;
    // Merge
    pred_chain->nodes.insert(pred_chain->nodes.end(), cur_chain->nodes.begin(),
                             cur_chain->nodes.end());
    pred_chain->ancestors_and_this.insert(cur_chain->nodes.begin(),
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
    if (cur_logi_node->op()->IsDecodeOp()) { continue; }
    if (cur_logi_node->op()->IsPrintOp()) { continue; }
    if (cur_logi_node->op()->IsRecurrentOp()) { continue; }
    if (cur_logi_node->shared_model_nodes()) { continue; }
    data_parallel_node.push_back(cur_logi_node);
  }
  while (DoOneDataMerge(data_parallel_node, chain_list, logical2chain_it)) {}
}

}  // namespace

ChainGraph::ChainGraph(bool is_train) {
  HashMap<ChainNode*, const LogicalNode*> chain2first_shared;
  BuildFwStruct(is_train, &chain2first_shared);
  BuildRecordLoadStruct();
  if (is_train) {
    BuildBwStruct();
    BuildLossPrintStruct();
  }
  BuildModelStruct(is_train, chain2first_shared);
  BuildRecurrentStruct();
  RemoveNeedlessCloneOp();
  ForEachNode([](ChainNode* node) { node->set_data_output_lbns(); });
  ToDotWithAutoFilePath();
}

void ChainGraph::BuildFwStruct(
    bool is_train,
    HashMap<ChainNode*, const LogicalNode*>* chain2first_shared) {
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
  HashMap<ChainNode*, HashSet<ChainNode*>> chain_node2pred;
  FOR_EACH(chain_it, chain_list) {
    ChainNode* chain_node = nullptr;
    if (chain_it->nodes.size() == 1) {
      std::shared_ptr<const Operator> op = chain_it->nodes[0]->op();
      if (op->IsLossOp() && is_train) {
        chain_node = NewNode<LossChainNode>();
      } else if (op->IsDecodeOp()) {
        chain_node = NewNode<DecodeChainNode>();
      } else if (op->IsPrintOp()) {
        chain_node = NewNode<PrintChainNode>();
      } else {
        // do nothing
      }
    } else if (chain_it->nodes[0]->op()->IsDecodeOp()) {
      chain_node = NewNode<DecodeChainNode>();
    }
    if (chain_node == nullptr) { chain_node = NewNode<ForwardChainNode>(); }
    chain_it2chain_node[chain_it] = chain_node;
    chain_node2pred[chain_node] = {};
    CHECK(!chain_it->nodes.empty());
    chain_node->mut_parallel_desc() = chain_it->nodes.front()->parallel_desc();
    for (const LogicalNode* logical_node : chain_it->nodes) {
      chain_node->mut_op_vec().push_back(logical_node->op());
      if (logical_node->shared_model_nodes()) {
        CHECK_EQ(chain_it->nodes.size(), 1);
        CHECK(
            chain2first_shared
                ->emplace(chain_node, logical_node->shared_model_nodes()->at(0))
                .second);
      }
    }
  }
  // Print the predecessor
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

void ChainGraph::BuildRecordLoadStruct() {
  HashMap<std::string, std::vector<DecodeChainNode*>> data_info2decode_nodes;
  HashMap<std::string, int32_t> data_info2suffix_length;
  ForEachChainNode<DecodeChainNode>([&](DecodeChainNode* decode_node) {
    std::shared_ptr<const Operator> decode_op = decode_node->SoleOp();
    std::string data_dir =
        decode_op->GetValFromCustomizedConf<std::string>("data_dir");
    std::string part_name_prefix =
        decode_op->GetValFromCustomizedConf<std::string>("part_name_prefix");
    std::string data_info = data_dir + "_" + part_name_prefix;
    data_info2decode_nodes[data_info].emplace_back(decode_node);
    int32_t part_name_suffix_length =
        decode_op->GetValFromCustomizedConf<int32_t>("part_name_suffix_length");
    if (data_info2suffix_length.find(data_info)
        != data_info2suffix_length.end()) {
      CHECK_EQ(data_info2suffix_length[data_info], part_name_suffix_length);
    } else {
      data_info2suffix_length[data_info] = part_name_suffix_length;
    }
  });
  for (auto& pair : data_info2decode_nodes) {
    std::vector<std::shared_ptr<const ParallelDesc>> parallel_descs;
    for (DecodeChainNode* decode_node : pair.second) {
      auto iter = std::find_if(
          parallel_descs.begin(), parallel_descs.end(),
          [&](std::shared_ptr<const ParallelDesc>& parallel_desc) {
            return parallel_desc->Equal(decode_node->parallel_desc().get());
          });
      if (iter == parallel_descs.end()) {
        parallel_descs.emplace_back(decode_node->parallel_desc());
      }
    }
    LOG_IF(WARNING, parallel_descs.size() > 1)
        << "Operators sharing same data information belong to different "
           "placement groups";
    for (auto parallel_desc : parallel_descs) {
      ChainNode* record_load_node = NewNode<RecordLoadChainNode>();
      record_load_node->mut_parallel_desc() = parallel_desc;
      for (DecodeChainNode* decode_node : pair.second) {
        if (!decode_node->parallel_desc()->Equal(parallel_desc.get())) {
          continue;
        }
        Connect<ChainNode>(record_load_node, NewEdge(), decode_node);
      }
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
    if (fw_src_node == nullptr) { continue; }  // SourceChainNode
    auto fw_dst_node = dynamic_cast<ForwardChainNode*>(fw_edge->dst_node());
    ChainNode* bw_src_node = fw_src_node->bw_node();
    if (bw_src_node == nullptr) { continue; }
    if (fw_dst_node == nullptr) {
      if (dynamic_cast<LossChainNode*>(fw_edge->dst_node())) {
        Connect(fw_edge->dst_node(), NewEdge(), bw_src_node);
      }
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

void ChainGraph::BuildLossPrintStruct() {
  ForEachChainNode<LossChainNode>([&](LossChainNode* loss_chain) {
    std::shared_ptr<const Operator> loss_op = loss_chain->SoleOp();
    // Reduce Sum op
    OperatorConf sum_op_conf;
    sum_op_conf.set_name("sum_op_" + NewUniqueId());
    sum_op_conf.mutable_reduce_sum_conf()->set_in(loss_op->Lbn4BnInOp("loss"));
    sum_op_conf.mutable_reduce_sum_conf()->set_out("out");
    sum_op_conf.mutable_reduce_sum_conf()->set_axis(0);
    std::shared_ptr<Operator> sum_op = ConstructOp(sum_op_conf);
    loss_chain->mut_op_vec().push_back(sum_op);
    // Loss Accumulate Chain
    OperatorConf loss_acc_op_conf;
    loss_acc_op_conf.set_name("loss_acc_" + NewUniqueId());
    loss_acc_op_conf.mutable_accumulate_conf();
    auto loss_acc_op = ConstructOp(loss_acc_op_conf);
    auto loss_acc_chain = NewNode<LossAccChainNode>();
    loss_acc_chain->mut_op_vec() = {loss_acc_op};
    loss_acc_chain->mut_parallel_desc() = loss_chain->parallel_desc();
    Connect<ChainNode>(loss_chain, NewEdge(), loss_acc_chain);
    // Loss Print Chain
    OperatorConf loss_print_op_conf;
    loss_print_op_conf.set_name("loss_print_" + loss_op->op_name());
    loss_print_op_conf.mutable_loss_print_conf();
    loss_print_op_conf.mutable_loss_print_conf()->set_loss_lbn(
        sum_op->Lbn4BnInOp("out"));
    if (!loss_op->GetValFromCustomizedConf<std::string>("weight").empty()) {
      loss_print_op_conf.mutable_loss_print_conf()->set_reduction_lbn(
          loss_op->Lbn4BnInOp("reduction_coefficient"));
    }
    loss_print_op_conf.mutable_loss_print_conf()->set_weight_scalar(
        loss_op->GetValFromCustomizedConf<float>("weight_scalar"));
    loss_print_op_conf.mutable_loss_print_conf()->set_reduction_type(
        static_cast<LossReductionType>(
            loss_op->GetEnumFromCustomizedConf("reduction")));
    auto loss_print_op = ConstructOp(loss_print_op_conf);
    ParallelConf loss_print_pr_conf;
    loss_print_pr_conf.set_policy(kDataParallel);
    loss_print_pr_conf.add_device_name(
        Global<IDMgr>::Get()->MachineName4MachineId(0) + ":cpu:1");
    auto loss_print_chain = NewNode<LossPrintChainNode>();
    loss_print_chain->mut_op_vec() = {loss_print_op};
    loss_print_chain->mut_parallel_desc().reset(
        new ParallelDesc(loss_print_pr_conf));
    Connect<ChainNode>(loss_acc_chain, NewEdge(), loss_print_chain);
  });
}

NormalMdUpdtChainNode* ChainGraph::BuildNormalMdUpdtAndMdSaveStruct(
    bool is_train, ForwardChainNode* fw_chain) {
  NormalMdUpdtChainNode* md_updt_chain = NewNode<NormalMdUpdtChainNode>();
  md_updt_chain->mut_parallel_desc() = fw_chain->parallel_desc();
  if (is_train) { BuildMdSaveStruct(fw_chain, md_updt_chain); }
  return md_updt_chain;
}

MdSaveChainNode* ChainGraph::BuildMdSaveStruct(const ForwardChainNode* fw_chain,
                                               ChainNode* need_save_chain) {
  OperatorConf md_save_op_conf;
  md_save_op_conf.set_name("md_save_" + NewUniqueId());
  auto model_save_op = ConstructOp(md_save_op_conf);
  auto md_save_chain = NewNode<MdSaveChainNode>();
  md_save_chain->mut_op_vec() = {model_save_op};
  auto md_save_pr_desc = new ParallelDesc(*(fw_chain->parallel_desc()));
  if (fw_chain->parallel_desc()->policy() == ParallelPolicy::kDataParallel) {
    md_save_pr_desc->RemoveNeedlessDevice(1);
  }
  md_save_pr_desc->set_device_type(DeviceType::kCPU);
  md_save_chain->mut_parallel_desc().reset(md_save_pr_desc);
  Connect<ChainNode>(need_save_chain, NewEdge(), md_save_chain);
  return md_save_chain;
}

void ChainGraph::BuildModelStruct(
    bool is_train,
    const HashMap<ChainNode*, const LogicalNode*>& chain2first_shared) {
  HashMap<const LogicalNode*, NormalMdUpdtChainNode*> first_shared2mdupdt;
  ForEachChainNode<ForwardChainNode>([&](ForwardChainNode* fw_chain) {
    if (is_train && fw_chain->HasOpWithForwardModelBlob()) {
      BuildMdSaveStruct(fw_chain, fw_chain);
    }
    if (fw_chain->HasOpWithModelOrModelTmpBlob()) {
      // MdUpdt MdSave
      NormalMdUpdtChainNode* md_updt_chain = nullptr;
      auto chain2first_shared_it = chain2first_shared.find(fw_chain);
      if (chain2first_shared_it == chain2first_shared.end()) {
        md_updt_chain = BuildNormalMdUpdtAndMdSaveStruct(is_train, fw_chain);
      } else {
        auto first_shared2mdupdt_it =
            first_shared2mdupdt.find(chain2first_shared_it->second);
        if (first_shared2mdupdt_it == first_shared2mdupdt.end()) {
          md_updt_chain = BuildNormalMdUpdtAndMdSaveStruct(is_train, fw_chain);
          CHECK(first_shared2mdupdt
                    .emplace(chain2first_shared_it->second, md_updt_chain)
                    .second);
        } else {
          md_updt_chain = first_shared2mdupdt_it->second;
        }
      }
      Connect<ChainNode>(md_updt_chain, NewEdge(), fw_chain);
      // Model Diff Accumulate Chain
      if (is_train && fw_chain->HasOpWithModelBlob()) {
        BackwardChainNode* bw_chain = fw_chain->bw_node();
        Connect<ChainNode>(md_updt_chain, NewEdge(), bw_chain);
        OperatorConf md_diff_acc_op_conf;
        md_diff_acc_op_conf.set_name("md_diff_acc_" + NewUniqueId());
        md_diff_acc_op_conf.mutable_accumulate_conf();
        auto md_diff_acc_op = ConstructOp(md_diff_acc_op_conf);
        auto md_diff_acc_chain = NewNode<MdDiffAccChainNode>();
        md_diff_acc_chain->mut_op_vec() = {md_diff_acc_op};
        auto md_diff_acc_pr_desc =
            new ParallelDesc(*(fw_chain->parallel_desc()));
        md_diff_acc_pr_desc->set_policy(kInvalidParallel);
        md_diff_acc_chain->mut_parallel_desc().reset(md_diff_acc_pr_desc);
        Connect<ChainNode>(bw_chain, NewEdge(), md_diff_acc_chain);
        Connect<ChainNode>(md_diff_acc_chain, NewEdge(), md_updt_chain);
      }
    }
  });
}

void ChainGraph::BuildRecurrentStruct() {
  ForEachNode([&](ChainNode* chain_node) {
    if (chain_node->HasSoleRecurrentOp()) {
      Connect(chain_node, NewEdge(), chain_node);
    }
  });
}

void ChainGraph::RemoveNeedlessCloneOp() {
  TopoForEachNode([&](ChainNode* chain_node) {
    HashMap<std::string, std::string> olbn2ilbn_in_clone_op;
    auto fw_chain_node = dynamic_cast<ForwardChainNode*>(chain_node);
    if (fw_chain_node == nullptr) { return; }
    for (std::shared_ptr<const Operator> op : fw_chain_node->op_vec()) {
      if (!op->IsCloneOp()) { continue; }
      const std::string& ilbn = op->Lbn4BnInOp(op->SoleIbn());
      for (const std::string& obn : op->output_bns()) {
        CHECK(olbn2ilbn_in_clone_op.emplace(op->Lbn4BnInOp(obn), ilbn).second);
      }
    }
    ModifyOpLbn4BnInChainNode(olbn2ilbn_in_clone_op, chain_node);
    fw_chain_node->ForEachNodeOnOutEdge([&](ChainNode* succ_chain_node) {
      ModifyOpLbn4BnInChainNode(olbn2ilbn_in_clone_op, succ_chain_node);
    });
    auto& op_vec_in_fw = fw_chain_node->mut_op_vec();
    Erase<std::vector<std::shared_ptr<Operator>>>(
        op_vec_in_fw,
        [&](const std::shared_ptr<Operator>& op) { return op->IsCloneOp(); });
  });
}

}  // namespace oneflow
