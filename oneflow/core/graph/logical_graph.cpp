#include "oneflow/core/graph/logical_graph.h"
#include "oneflow/core/graph/task_graph.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/operator/op_conf.pb.h"

namespace oneflow {

LogicalGraph::LogicalGraph(bool is_train) {
  HashMap<LogicalEdge*, std::string> edge2ibn;
  BuildFwStruct(&edge2ibn);
  SetMainModelParallel();
  if (is_train) { BuildBwStruct(&edge2ibn); }
  MergeEdge();
  SetNodeDataLbi();
  if (is_train) { BuildLossPrintStruct(); }
  BuildModelStruct(is_train);
  BuildRecordLoadStruct();
  if (is_train) { ConnectFwToBw(); }
  ToDotWithAutoFilePath();
}

template<typename LogicalNodeType>
void LogicalGraph::ForEachLogicalNode(std::function<void(LogicalNodeType*)> func) {
  std::vector<LogicalNodeType*> valid_nodes;
  ForEachNode([&](LogicalNode* logical_node) {
    auto valid_node = dynamic_cast<LogicalNodeType*>(logical_node);
    if (valid_node != nullptr) { valid_nodes.push_back(valid_node); }
  });
  for (LogicalNodeType* valid_node : valid_nodes) { func(valid_node); }
}

void LogicalGraph::BuildFwStruct(HashMap<LogicalEdge*, std::string>* edge2ibn) {
  HashMap<std::string, std::vector<LogicalNode*>> op_name2nodes;
  NaiveBuildFwStruct(edge2ibn, &op_name2nodes);
  FixSharedModelNodes(op_name2nodes);
  AddB121Clone(edge2ibn);
  total_mbn_num_ = 0;
  ForEachNode([&](LogicalNode* node) {
    total_mbn_num_ +=
        node->SoleOp()->model_bns().size() + node->SoleOp()->forward_model_bns().size();
  });
}

void LogicalGraph::NaiveBuildFwStruct(
    HashMap<LogicalEdge*, std::string>* edge2ibn,
    HashMap<std::string, std::vector<LogicalNode*>>* op_name2nodes) {
  const DLNetConf& dlnet_conf = Global<JobDesc>::Get()->dlnet_conf();
  const Placement& placement = Global<JobDesc>::Get()->placement();
  HashMap<std::string, ParallelDesc*> name2parallel_desc;
  for (const PlacementGroup& p_group : placement.placement_group()) {
    for (const std::string& op_name : p_group.op_set().op_name()) {
      auto parallel_desc_raw_ptr = new ParallelDesc(p_group.parallel_conf());
      CHECK(name2parallel_desc.emplace(op_name, parallel_desc_raw_ptr).second);
    }
  }

  HashMap<LogicalBlobId, LogicalNode*> lbi2producer;
  for (OperatorConf cur_op_conf : dlnet_conf.op()) {
    ParallelDesc* parallel_desc_raw_ptr = name2parallel_desc.at(cur_op_conf.name());
    cur_op_conf.set_device_type(parallel_desc_raw_ptr->device_type());
    std::shared_ptr<Operator> cur_op = ConstructOp(cur_op_conf);
    LogicalNode* cur_node = cur_op->NewProperLogicalNode();
    AddAllocatedNode(cur_node);
    cur_node->mut_op_vec() = {cur_op};
    cur_node->SoleOp()->FixParallelDesc(parallel_desc_raw_ptr);
    cur_node->mut_parallel_desc().reset(parallel_desc_raw_ptr);
    for (const std::string& obn : cur_node->SoleOp()->output_bns()) {
      const LogicalBlobId& lbi = cur_node->SoleOp()->BnInOp2Lbi(obn);
      CHECK(lbi2producer.emplace(lbi, cur_node).second);
    }
    (*op_name2nodes)[cur_op->op_name()].push_back(cur_node);
  }
  ForEachNode([&](LogicalNode* cur_node) {
    for (const std::string& ibn : cur_node->SoleOp()->input_bns()) {
      const LogicalBlobId& lbi = cur_node->SoleOp()->BnInOp2Lbi(ibn);
      LogicalNode* pred_node = lbi2producer.at(lbi);
      if (pred_node == cur_node) { continue; }
      LogicalEdge* edge = NewEdge();
      edge->mut_lbis() = {lbi};
      CHECK(edge2ibn->emplace(edge, ibn).second);
      Connect(pred_node, edge, cur_node);
    }
  });
}

void LogicalGraph::FixSharedModelNodes(
    const HashMap<std::string, std::vector<LogicalNode*>>& op_name2nodes) {
  const DLNetConf& dlnet_conf = Global<JobDesc>::Get()->dlnet_conf();
  for (const OpNameSet& op_name_set : dlnet_conf.shared_model_group()) {
    auto shared_model_nodes = std::make_shared<std::vector<LogicalNode*>>();
    for (const std::string& op_name : op_name_set.op_name()) {
      CHECK_EQ(op_name2nodes.at(op_name).size(), 1);
      shared_model_nodes->push_back(op_name2nodes.at(op_name).front());
    }
    SortAndRemoveDuplication(shared_model_nodes.get());
    for (LogicalNode* cur_node : *shared_model_nodes) {
      cur_node->mut_shared_model_nodes() = shared_model_nodes;
    }
    const std::string& shared_op_name = shared_model_nodes->front()->SoleOp()->op_name();
    FOR_RANGE(size_t, i, 1, shared_model_nodes->size()) {
      shared_model_nodes->at(i)->SoleOp()->FixLbiWhenShareModel(shared_op_name);
    }
  }
  ForEachNode([&](LogicalNode* cur_node) {
    if (cur_node->shared_model_nodes()) {
      for (LogicalNode* shared_node : *(cur_node->shared_model_nodes())) {
        if (shared_node->parallel_desc() == nullptr) { continue; }
        if (cur_node->parallel_desc()) {
          CHECK(cur_node->parallel_desc()->Equal(shared_node->parallel_desc().get()));
        } else {
          cur_node->mut_parallel_desc() = shared_node->parallel_desc();
        }
      }
    } else {
      // do nothing
    }
    CHECK(cur_node->parallel_desc())
        << "Please set the placement of " << cur_node->SoleOp()->op_name();
  });
}

void LogicalGraph::AddB121Clone(HashMap<LogicalEdge*, std::string>* edge2ibn) {
  std::vector<B121CloneInfo> clone_infos;
  CollectB121CloneInfos(&clone_infos);
  for (const B121CloneInfo& clone_info : clone_infos) { AddOneB121CloneNode(clone_info, edge2ibn); }
}

void LogicalGraph::CollectB121CloneInfos(std::vector<B121CloneInfo>* clone_infos) {
  ForEachNode([&](LogicalNode* cur_node) {
    HashMap<LogicalBlobId, B121CloneInfo> lbi2clone_info;
    for (LogicalEdge* edge : cur_node->out_edges()) {
      B121CloneInfo& clone_info = lbi2clone_info[edge->SoleLbi()];
      BldSubTskGphMthd mthd = GetMthdForBldSubTskGph(cur_node, edge->dst_node());
      if (mthd == &TaskGraph::BldSubTskGphByBoxing) {
        clone_info.edges_boxing.push_back(edge);
      } else if (mthd == &TaskGraph::BldSubTskGphByOneToOne) {
        clone_info.edges_121.push_back(edge);
      } else {
        UNIMPLEMENTED();
      }
    }
    for (auto& pair : lbi2clone_info) {
      if (pair.second.edges_boxing.empty()) { continue; }
      if (pair.second.edges_121.empty()) { continue; }
      pair.second.pred_node = cur_node;
      pair.second.lbi = pair.first;
      clone_infos->push_back(pair.second);
    }
  });
}

void LogicalGraph::AddOneB121CloneNode(const B121CloneInfo& clone_info,
                                       HashMap<LogicalEdge*, std::string>* edge2ibn) {
  // lbi_boxing, lbi_121
  LogicalBlobId lbi_boxing = clone_info.lbi;
  lbi_boxing.set_b121_id(0);
  LogicalBlobId lbi_121 = clone_info.lbi;
  lbi_121.set_b121_id(1);
  // Clone Op
  OperatorConf clone_op_conf;
  clone_op_conf.set_name("b121_clone_" + NewUniqueId());
  clone_op_conf.set_device_type(clone_info.pred_node->SoleOp()->device_type());
  clone_op_conf.mutable_clone_conf()->set_out_num(2);
  std::shared_ptr<Operator> clone_op = ConstructOp(clone_op_conf);
  *(clone_op->MutBnInOp2Lbi(clone_op->SoleIbn())) = clone_info.lbi;
  *(clone_op->MutBnInOp2Lbi(clone_op->SoleIdbn())) = clone_info.lbi;
  *(clone_op->MutBnInOp2Lbi(clone_op->output_bns().Get(0))) = lbi_boxing;
  *(clone_op->MutBnInOp2Lbi(clone_op->output_diff_bns().Get(0))) = lbi_boxing;
  *(clone_op->MutBnInOp2Lbi(clone_op->output_bns().Get(1))) = lbi_121;
  *(clone_op->MutBnInOp2Lbi(clone_op->output_diff_bns().Get(1))) = lbi_121;
  // Clone LogicalNode
  LogicalNode* clone_node = NewNode<NormalForwardLogicalNode>();
  clone_node->mut_op_vec() = {clone_op};
  clone_node->mut_parallel_desc() = clone_info.pred_node->parallel_desc();
  // Connect
  LogicalEdge* edge = NewEdge();
  edge->mut_lbis() = {clone_info.lbi};
  Connect(clone_info.pred_node, edge, clone_node);
  CHECK(edge2ibn->emplace(edge, clone_op->SoleIbn()).second);
  ReConnectToFwClone(clone_node, lbi_boxing, clone_info.edges_boxing, *edge2ibn);
  ReConnectToFwClone(clone_node, lbi_121, clone_info.edges_121, *edge2ibn);
}

void LogicalGraph::ReConnectToFwClone(LogicalNode* clone_node, const LogicalBlobId& lbi,
                                      const std::vector<LogicalEdge*>& edges,
                                      const HashMap<LogicalEdge*, std::string>& edge2ibn) {
  for (LogicalEdge* edge : edges) {
    LogicalNode* dst_node = edge->dst_node();
    const std::string& ibn = edge2ibn.at(edge);
    *(dst_node->SoleOp()->MutBnInOp2Lbi(ibn)) = lbi;
    *(dst_node->SoleOp()->MutBnInOp2Lbi(GenDiffBn(ibn))) = lbi;
    DisConnect(edge);
    Connect(clone_node, edge, dst_node);
    edge->mut_lbis() = {lbi};
  }
}

void LogicalGraph::SetMainModelParallel() {
  ForEachNode([](LogicalNode* node) {
    if (node->parallel_desc()->policy() == kModelParallel) { node->set_main_model_parallel(node); }
  });
  ForEachNode([](LogicalNode* node) {
    LogicalNode* pred_node = node;
    while (pred_node->SoleOp()->IsElemWiseOp()) { pred_node = pred_node->SoleInEdge()->src_node(); }
    if (pred_node != node && pred_node->parallel_desc()->policy() == kModelParallel) {
      node->mut_parallel_desc() = pred_node->parallel_desc();
      node->set_main_model_parallel(pred_node);
    }
  });
}

void LogicalGraph::BuildBwStruct(HashMap<LogicalEdge*, std::string>* edge2ibn) {
  NaiveBuildBwStruct(edge2ibn);
  AddBackwardClone(*edge2ibn);
}

void LogicalGraph::NaiveBuildBwStruct(HashMap<LogicalEdge*, std::string>* edge2ibn) {
  HashSet<LogicalNode*> nodes_need_bw;
  TopoForEachNode([&](LogicalNode* logical_node) {
    auto fw_node = dynamic_cast<ForwardLogicalNode*>(logical_node);
    if (fw_node == nullptr) { return; }
    if (fw_node->HasOpWithModelBlob()) {
      CHECK(nodes_need_bw.insert(fw_node).second);
      return;
    }
    for (LogicalEdge* edge : fw_node->in_edges()) {
      if (nodes_need_bw.find(edge->src_node()) != nodes_need_bw.end()) {
        CHECK(nodes_need_bw.insert(fw_node).second);
        return;
      }
    }
  });
  for (LogicalNode* fw_node : nodes_need_bw) {
    BackwardLogicalNode* bw_node = static_cast<ForwardLogicalNode*>(fw_node)->NewBackwardNode();
    if (bw_node) { AddAllocatedNode(bw_node); }
  }
  std::list<LogicalEdge*> fw_edges;
  ForEachEdge([&](LogicalEdge* edge) { fw_edges.push_back(edge); });
  for (LogicalEdge* fw_edge : fw_edges) {
    auto NewBwEdge = [&]() {
      LogicalEdge* bw_edge = NewEdge();
      bw_edge->mut_lbis() = fw_edge->lbis();
      CHECK(edge2ibn->emplace(bw_edge, edge2ibn->at(fw_edge)).second);
      return bw_edge;
    };
    auto fw_src_node = dynamic_cast<ForwardLogicalNode*>(fw_edge->src_node());
    if (fw_src_node == nullptr) { continue; }
    BackwardLogicalNode* bw_src_node = fw_src_node->bw_node();
    if (bw_src_node == nullptr) { continue; }
    auto fw_dst_node = dynamic_cast<ForwardLogicalNode*>(fw_edge->dst_node());
    if (fw_dst_node == nullptr) {
      if (dynamic_cast<LossLogicalNode*>(fw_edge->dst_node())) {
        Connect<LogicalNode>(fw_edge->dst_node(), NewBwEdge(), bw_src_node);
      }
    } else {
      BackwardLogicalNode* bw_dst_node = fw_dst_node->bw_node();
      if (bw_dst_node) { Connect<LogicalNode>(bw_dst_node, NewBwEdge(), bw_src_node); }
    }
  }
}

void LogicalGraph::AddBackwardClone(const HashMap<LogicalEdge*, std::string>& edge2ibn) {
  std::vector<BackwardCloneInfo> clone_infos;
  ForEachLogicalNode<BackwardLogicalNode>([&](BackwardLogicalNode* bw_node) {
    HashMap<LogicalBlobId, BackwardCloneInfo> lbi2clone_info;
    for (LogicalEdge* edge : bw_node->in_edges()) {
      lbi2clone_info[edge->SoleLbi()].edges.push_back(edge);
    }
    for (auto& pair : lbi2clone_info) {
      BackwardCloneInfo& clone_info = pair.second;
      if (clone_info.edges.size() <= 1) { continue; }
      clone_info.succ_node = bw_node;
      clone_info.lbi = pair.first;
      clone_infos.push_back(clone_info);
    }
  });
  for (const BackwardCloneInfo& clone_info : clone_infos) {
    AddOneBackwardClone(clone_info, edge2ibn);
  }
}

void LogicalGraph::AddOneBackwardClone(const BackwardCloneInfo& clone_info,
                                       const HashMap<LogicalEdge*, std::string>& edge2ibn) {
  OperatorConf clone_op_conf;
  clone_op_conf.set_name("bw_clone_" + NewUniqueId());
  clone_op_conf.set_device_type(clone_info.succ_node->SoleOp()->device_type());
  clone_op_conf.mutable_clone_conf()->set_out_num(clone_info.edges.size());
  std::shared_ptr<Operator> clone_op = ConstructOp(clone_op_conf);
  *(clone_op->MutBnInOp2Lbi(clone_op->SoleIdbn())) = clone_info.lbi;
  FOR_RANGE(size_t, i, 0, clone_info.edges.size()) {
    LogicalBlobId lbi_clone_i = clone_info.lbi;
    lbi_clone_i.set_clone_id(i);
    *(clone_op->MutBnInOp2Lbi(clone_op->output_diff_bns().Get(i))) = lbi_clone_i;
  }
  LogicalNode* clone_node = NewNode<NormalBackwardLogicalNode>();
  clone_node->mut_op_vec() = {clone_op};
  clone_node->mut_parallel_desc() = clone_info.succ_node->parallel_desc();
  LogicalEdge* clone_in_diff_edge = NewEdge();
  clone_in_diff_edge->mut_lbis() = {clone_info.lbi};
  Connect(clone_node, clone_in_diff_edge, clone_info.succ_node);
  FOR_RANGE(size_t, i, 0, clone_op->output_diff_bns().size()) {
    const LogicalBlobId& lbi_clone_i = clone_op->BnInOp2Lbi(clone_op->output_diff_bns().Get(i));
    LogicalEdge* edge = clone_info.edges.at(i);
    std::vector<LogicalBlobId>& edge_lbis = edge->mut_lbis();
    CHECK_EQ(1, edge_lbis.size());
    edge_lbis.front() = lbi_clone_i;
    LogicalNode* src_node = edge->src_node();
    *(src_node->SoleOp()->MutBnInOp2Lbi(GenDiffBn(edge2ibn.at(edge)))) = lbi_clone_i;
    DisConnect(edge);
    Connect(src_node, edge, clone_node);
  }
}

void LogicalGraph::MergeEdge() {
  ForEachNode([](LogicalNode* node) {
    HashMap<LogicalNode*, std::vector<LogicalEdge*>> dst2edges;
    for (LogicalEdge* out_edge : node->out_edges()) {
      dst2edges[out_edge->dst_node()].push_back(out_edge);
    }
    for (const auto& pair : dst2edges) {
      std::vector<LogicalBlobId>& lbi_all = pair.second.at(0)->mut_lbis();
      FOR_RANGE(size_t, i, 1, pair.second.size()) {
        std::vector<LogicalBlobId>& lbi_i = pair.second.at(i)->mut_lbis();
        lbi_all.insert(lbi_all.end(), lbi_i.begin(), lbi_i.end());
        lbi_i.clear();
        DisConnect(pair.second.at(i));  // TODO: delete its memory ?
      }
    }
  });
}

void LogicalGraph::SetNodeDataLbi() {
  ForEachNode([](LogicalNode* node) {
    for (LogicalEdge* out_edge : node->out_edges()) {
      node->SetDataLbisTo(out_edge->dst_node(), out_edge->lbis());
    }
  });
}

void LogicalGraph::BuildLossPrintStruct() {
  ForEachLogicalNode<LossLogicalNode>([&](LossLogicalNode* loss_logical) {
    std::shared_ptr<const Operator> loss_op = loss_logical->SoleOp();
    // Reduce Sum op
    OperatorConf reduce_loss_op_conf;
    reduce_loss_op_conf.set_name("reduce_loss_" + loss_op->op_name());
    reduce_loss_op_conf.set_device_type(loss_op->device_type());
    auto reduce_sum_conf = reduce_loss_op_conf.mutable_reduce_sum_conf();
    *(reduce_sum_conf->mutable_in_sys()) = loss_op->BnInOp2Lbi("loss");
    reduce_sum_conf->set_out("out");
    std::shared_ptr<Operator> reduce_loss_op = ConstructOp(reduce_loss_op_conf);
    loss_logical->mut_op_vec().push_back(reduce_loss_op);
    // Loss Accumulate Logical
    OperatorConf loss_acc_op_conf;
    loss_acc_op_conf.set_name("loss_acc_" + loss_op->op_name());
    loss_acc_op_conf.set_device_type(loss_op->device_type());
    loss_acc_op_conf.mutable_accumulate_conf();
    std::shared_ptr<Operator> loss_acc_op = ConstructOp(loss_acc_op_conf);
    LossAccLogicalNode* loss_acc_logical = NewNode<LossAccLogicalNode>();
    loss_acc_logical->mut_op_vec() = {loss_acc_op};
    loss_acc_logical->mut_parallel_desc() = loss_logical->parallel_desc();
    Connect<LogicalNode>(loss_logical, NewEdge(), loss_acc_logical);
    // Loss Print Logical
    OperatorConf loss_print_op_conf;
    loss_print_op_conf.set_name("loss_print_" + loss_op->op_name());
    loss_print_op_conf.set_device_type(DeviceType::kCPU);
    auto loss_print_conf = loss_print_op_conf.mutable_loss_print_conf();

    *(loss_print_conf->mutable_loss_lbi()) = reduce_loss_op->BnInOp2Lbi("out");

    if (!loss_op->GetValFromCustomizedConf<std::string>("weight").empty()) {
      *(loss_print_conf->mutable_reduction_lbi()) = loss_op->BnInOp2Lbi("reduction_coefficient");
    }
    loss_print_conf->set_weight_scalar(loss_op->GetValFromCustomizedConf<float>("weight_scalar"));
    loss_print_conf->set_reduction_type(
        static_cast<LossReductionType>(loss_op->GetEnumFromCustomizedConf("reduction")));
    std::shared_ptr<Operator> loss_print_op = ConstructOp(loss_print_op_conf);
    ParallelConf loss_print_pr_conf;
    loss_print_pr_conf.set_policy(kDataParallel);
    loss_print_pr_conf.add_device_name(Global<IDMgr>::Get()->MachineName4MachineId(0) + ":cpu:1");
    LossPrintLogicalNode* loss_print_logical = NewNode<LossPrintLogicalNode>();
    loss_print_logical->mut_op_vec() = {loss_print_op};
    loss_print_logical->mut_parallel_desc().reset(new ParallelDesc(loss_print_pr_conf));
    Connect<LogicalNode>(loss_acc_logical, NewEdge(), loss_print_logical);
  });
}

void LogicalGraph::BuildModelStruct(bool is_train) {
  HashMap<const LogicalNode*, NormalMdUpdtLogicalNode*> first_shared2mdupdt;
  ForEachLogicalNode<ForwardLogicalNode>([&](ForwardLogicalNode* fw_logical) {
    if (is_train && fw_logical->HasOpWithForwardModelBlob()) {
      BuildMdSaveStruct(fw_logical, fw_logical);
    }
    if (fw_logical->HasOpWithModelOrConstModelBlob()) {
      // MdUpdt MdSave
      NormalMdUpdtLogicalNode* md_updt_logical = nullptr;
      if (fw_logical->shared_model_nodes() == nullptr) {
        md_updt_logical = BuildNormalMdUpdtAndMdSaveStruct(is_train, fw_logical);
      } else {
        auto first_shared2mdupdt_it =
            first_shared2mdupdt.find(fw_logical->shared_model_nodes()->front());
        if (first_shared2mdupdt_it == first_shared2mdupdt.end()) {
          md_updt_logical = BuildNormalMdUpdtAndMdSaveStruct(is_train, fw_logical);
          CHECK(first_shared2mdupdt
                    .emplace(fw_logical->shared_model_nodes()->front(), md_updt_logical)
                    .second);
        } else {
          md_updt_logical = first_shared2mdupdt_it->second;
        }
      }
      Connect<LogicalNode>(md_updt_logical, NewEdge(), fw_logical);
      // Model Diff Accumulate Logical
      if (is_train && fw_logical->HasOpWithModelBlob()) {
        BackwardLogicalNode* bw_logical = fw_logical->bw_node();
        Connect<LogicalNode>(md_updt_logical, NewEdge(), bw_logical);
        LogicalNode* md_diff_acc_logical = nullptr;
        if (Global<JobDesc>::Get()->NumOfPiecesInBatch() > 1) {
          OperatorConf md_diff_acc_op_conf;
          md_diff_acc_op_conf.set_name("md_diff_acc_" + NewUniqueId());
          md_diff_acc_op_conf.set_device_type(fw_logical->parallel_desc()->device_type());
          md_diff_acc_op_conf.mutable_accumulate_conf();
          auto md_diff_acc_op = ConstructOp(md_diff_acc_op_conf);
          md_diff_acc_logical = NewNode<MdDiffAccLogicalNode>();
          md_diff_acc_logical->mut_op_vec() = {md_diff_acc_op};
          auto md_diff_acc_pr_desc = new ParallelDesc(*(fw_logical->parallel_desc()));
          md_diff_acc_logical->mut_parallel_desc().reset(md_diff_acc_pr_desc);
          Connect<LogicalNode>(bw_logical, NewEdge(), md_diff_acc_logical);
        } else {
          md_diff_acc_logical = bw_logical;
        }
        if (md_diff_acc_logical->parallel_desc()->parallel_num() > 1
            && md_diff_acc_logical->parallel_desc()->policy() == kDataParallel) {
          BuildReduceStruct(md_diff_acc_logical, md_updt_logical);
        } else {
          Connect<LogicalNode>(md_diff_acc_logical, NewEdge(), md_updt_logical);
        }
      }
    }
  });
  SetupNormalMdUpdtOp();
}

void LogicalGraph::BuildReduceStruct(LogicalNode* src, LogicalNode* dst) {
  std::shared_ptr<const ParallelDesc> src_pd = src->parallel_desc();
  std::shared_ptr<const ParallelDesc> dst_pd = dst->parallel_desc();
  CHECK_EQ(src_pd->parallel_num(), dst_pd->parallel_num());
  CHECK_EQ(src_pd->device_type(), dst_pd->device_type());
  // Reduce Scatter
  OperatorConf reduce_scatter_op_conf;
  reduce_scatter_op_conf.set_name("reduce_scatter_" + NewUniqueId());
  reduce_scatter_op_conf.set_device_type(src_pd->device_type());
  reduce_scatter_op_conf.mutable_reduce_scatter_conf()->set_out_num(src_pd->parallel_num());
  LogicalNode* reduce_scatter_node = NewNode<ReduceScatterLogicalNode>();
  reduce_scatter_node->mut_op_vec() = {ConstructOp(reduce_scatter_op_conf)};
  reduce_scatter_node->mut_parallel_desc() = src_pd;
  // Reduce Add
  OperatorConf reduce_add_op_conf;
  reduce_add_op_conf.set_name("reduce_add_" + NewUniqueId());
  reduce_add_op_conf.set_device_type(src_pd->device_type());
  reduce_add_op_conf.mutable_reduce_add_conf()->set_in_num(src_pd->parallel_num());
  LogicalNode* reduce_add_node = NewNode<ReduceAddLogicalNode>();
  reduce_add_node->mut_op_vec() = {ConstructOp(reduce_add_op_conf)};
  reduce_add_node->mut_parallel_desc() = src_pd;
  // Reduce Gather
  OperatorConf reduce_gather_op_conf;
  reduce_gather_op_conf.set_name("reduce_gather_" + NewUniqueId());
  reduce_gather_op_conf.set_device_type(src_pd->device_type());
  reduce_gather_op_conf.mutable_reduce_gather_conf()->set_in_num(src_pd->parallel_num());
  LogicalNode* reduce_gather_node = NewNode<ReduceGatherLogicalNode>();
  reduce_gather_node->mut_op_vec() = {ConstructOp(reduce_gather_op_conf)};
  reduce_gather_node->mut_parallel_desc() = src_pd;
  // Connect
  Connect(src, NewEdge(), reduce_scatter_node);
  Connect(reduce_scatter_node, NewEdge(), reduce_add_node);
  Connect(reduce_add_node, NewEdge(), reduce_gather_node);
  Connect(reduce_gather_node, NewEdge(), dst);
}

void LogicalGraph::SetupNormalMdUpdtOp() {
  ForEachLogicalNode<NormalMdUpdtLogicalNode>([](NormalMdUpdtLogicalNode* node) {
    if (node->in_edges().size() < 1) { return; }
    OperatorConf op_conf;
    op_conf.set_name("md_update_" + NewUniqueId());
    op_conf.set_device_type(node->parallel_desc()->device_type());
    NormalModelUpdateOpConf* mdupdt_conf = op_conf.mutable_normal_mdupdt_conf();
    const JobDesc* job_desc = Global<JobDesc>::Get();
    if (Global<JobDesc>::Get()->IsTrain()) {
      *(mdupdt_conf->mutable_user_conf()) = job_desc->other_conf().train_conf().model_update_conf();
    }
    mdupdt_conf->set_in_num(node->in_edges().size());
    node->mut_op_vec() = {ConstructOp(op_conf)};
  });
}

MdSaveLogicalNode* LogicalGraph::BuildMdSaveStruct(const ForwardLogicalNode* fw_logical,
                                                   LogicalNode* need_save_logical) {
  OperatorConf md_save_op_conf;
  md_save_op_conf.set_name("md_save_" + NewUniqueId());
  md_save_op_conf.set_device_type(fw_logical->parallel_desc()->device_type());
  md_save_op_conf.mutable_model_save_conf();
  auto model_save_op = ConstructOp(md_save_op_conf);
  auto md_save_logical = NewNode<MdSaveLogicalNode>();
  md_save_logical->mut_op_vec() = {model_save_op};
  auto md_save_pr_desc = new ParallelDesc(*(fw_logical->parallel_desc()));
  if (fw_logical->parallel_desc()->policy() == ParallelPolicy::kDataParallel) {
    md_save_pr_desc->RandomSelectOneDeviceAndRemoveTheOthers();
  }
  md_save_pr_desc->set_device_type(DeviceType::kCPU);
  md_save_logical->mut_parallel_desc().reset(md_save_pr_desc);
  Connect<LogicalNode>(need_save_logical, NewEdge(), md_save_logical);
  return md_save_logical;
}

NormalMdUpdtLogicalNode* LogicalGraph::BuildNormalMdUpdtAndMdSaveStruct(
    bool is_train, ForwardLogicalNode* fw_logical) {
  NormalMdUpdtLogicalNode* md_updt_logical = NewNode<NormalMdUpdtLogicalNode>();
  md_updt_logical->mut_parallel_desc() = fw_logical->parallel_desc();
  if (is_train) { BuildMdSaveStruct(fw_logical, md_updt_logical); }
  return md_updt_logical;
}

void LogicalGraph::BuildRecordLoadStruct() {
  HashMap<std::string, std::vector<DecodeLogicalNode*>> data_info2decode_nodes;
  HashMap<std::string, int32_t> data_info2suffix_length;
  ForEachLogicalNode<DecodeLogicalNode>([&](DecodeLogicalNode* decode_node) {
    std::shared_ptr<const Operator> decode_op = decode_node->SoleOp();
    if (decode_op->HasFieldInCustomizedConf("data_dir") == false) { return; }
    std::string data_dir = decode_op->GetValFromCustomizedConf<std::string>("data_dir");
    std::string part_name_prefix =
        decode_op->GetValFromCustomizedConf<std::string>("part_name_prefix");
    std::string data_info = data_dir + "_" + part_name_prefix;
    data_info2decode_nodes[data_info].emplace_back(decode_node);
    int32_t part_name_suffix_length =
        decode_op->GetValFromCustomizedConf<int32_t>("part_name_suffix_length");
    if (data_info2suffix_length.find(data_info) != data_info2suffix_length.end()) {
      CHECK_EQ(data_info2suffix_length[data_info], part_name_suffix_length);
    } else {
      data_info2suffix_length[data_info] = part_name_suffix_length;
    }
  });
  for (auto& pair : data_info2decode_nodes) {
    std::vector<std::shared_ptr<const ParallelDesc>> parallel_descs;
    for (DecodeLogicalNode* decode_node : pair.second) {
      auto iter = std::find_if(parallel_descs.begin(), parallel_descs.end(),
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
    for (std::shared_ptr<const ParallelDesc> parallel_desc : parallel_descs) {
      LogicalNode* record_load_node = NewNode<RecordLoadLogicalNode>();
      record_load_node->mut_parallel_desc() = parallel_desc;
      for (DecodeLogicalNode* decode_node : pair.second) {
        if (!decode_node->parallel_desc()->Equal(parallel_desc.get())) { continue; }
        Connect<LogicalNode>(record_load_node, NewEdge(), decode_node);
      }
    }
  }
}

void LogicalGraph::ConnectFwToBw() {
  ForEachLogicalNode<BackwardLogicalNode>([this](BackwardLogicalNode* bw_node) {
    if (bw_node->fw_node() == nullptr) { return; }
    Connect<LogicalNode>(bw_node->fw_node(), NewEdge(), bw_node);
  });
}

}  // namespace oneflow
