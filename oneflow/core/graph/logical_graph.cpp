#include "oneflow/core/graph/logical_graph.h"
#include "oneflow/core/graph/task_graph.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

namespace {

std::function<bool(const LogicalNode*)> MakePredicatorHasActualOutDiff(const LogicalGraph* graph) {
  std::list<LogicalNode*> loss_nodes;
  graph->ForEachNode([&](LogicalNode* node) {
    if (dynamic_cast<LossLogicalNode*>(node)) { loss_nodes.push_back(node); }
  });
  auto nodes_have_actual_out_diff_ptr = std::make_shared<HashSet<const LogicalNode*>>();
  auto HasBwConnection = [](const LogicalNode* prev, const LogicalNode* next) {
    HashSet<LogicalBlobId> idbn_lbis;
    for (const auto& idbn : next->SoleOp()->input_diff_bns()) {
      idbn_lbis.insert(next->SoleOp()->BnInOp2Lbi(idbn));
    }
    for (const auto& odbn : prev->SoleOp()->output_diff_bns()) {
      LogicalBlobId lbi = prev->SoleOp()->BnInOp2Lbi(odbn);
      if (idbn_lbis.find(lbi) != idbn_lbis.end()) { return true; }
    }
    return false;
  };
  auto ForEachNext = [&](LogicalNode* node, const std::function<void(LogicalNode*)>& Handler) {
    node->ForEachNodeOnInEdge([&](LogicalNode* in_node) {
      if (HasBwConnection(in_node, node)) { Handler(in_node); }
    });
  };
  graph->BfsForEachNode(loss_nodes, ForEachNext,
                        [nodes_have_actual_out_diff_ptr](LogicalNode* node) {
                          nodes_have_actual_out_diff_ptr->insert(node);
                        });
  return [nodes_have_actual_out_diff_ptr](const LogicalNode* node) {
    return nodes_have_actual_out_diff_ptr->find(node) != nodes_have_actual_out_diff_ptr->end();
  };
}

}  // namespace

LogicalGraph::LogicalGraph(bool is_train) {
  BuildFwStruct();
  if (is_train) { GroupNodesForReduceStruct(); }
  SetMainModelParallel();
  if (is_train) { BuildBwStruct(); }
  MergeEdge();
  SetNodeDataLbi();
  if (is_train) {
    BuildLossPrintStruct();
    BuildAccuracyPrintStruct();
  }
  BuildModelStruct(is_train);
  if (is_train) { ConnectFwToBw(); }
  ToDotWithAutoFilePath();
}

void LogicalGraph::GroupNodesForReduceStruct() {
  // get op model size
  HashMap<std::string, size_t> op_name2model_size;
  auto OpName2ModelSize = [&](const std::string& op_name) -> size_t {
    if (op_name2model_size.find(op_name) == op_name2model_size.end()) { return 0; }
    return op_name2model_size.at(op_name);
  };
  const JobDesc* global_job_desc = Global<JobDesc>::Get();
  OpGraph(global_job_desc).InferOpModelSize(&op_name2model_size);
  size_t model_total_size = 0;
  for (const auto& pair : op_name2model_size) { model_total_size += pair.second; }
  HashMap<ParallelDesc, std::list<const LogicalNode*>> parellel_desc2fw_group;
  size_t avg_size = model_total_size / global_job_desc->reduce_group_num();
  auto GetCurGroupSize = [&](int32_t group_id) {
    const size_t group_min_size = 16 * 1024 * 1024;
    size_t group_size = 0;
    if (group_id < global_job_desc->reduce_group_num()) {
      group_size = avg_size / (global_job_desc->reduce_group_num() - group_id);
    } else {
      group_size = avg_size;
    }
    return std::max(group_min_size, group_size);
  };
  // group fw nodes by parallel desc
  ReverseTopoForEachNode([&](LogicalNode* fw_node) {
    parellel_desc2fw_group[*fw_node->parallel_desc()].push_front(fw_node);
  });
  CHECK_GT(parellel_desc2fw_group.size(), 0);
  for (auto& pair : parellel_desc2fw_group) {
    fw_node_groups_.emplace_back(std::vector<const LogicalNode*>());
    auto& fw_node_group = pair.second;
    size_t cur_group_model_size = 0;
    int32_t group_id = 0;
    for (const LogicalNode* fw_node : fw_node_group) {
      fw_node_groups_.back().emplace_back(fw_node);
      cur_group_model_size += OpName2ModelSize(fw_node->SoleOp()->op_name());
      if (cur_group_model_size >= GetCurGroupSize(group_id)) {
        fw_node_groups_.emplace_back(std::vector<const LogicalNode*>());
        cur_group_model_size = 0;
        ++group_id;
      }
    }
  }
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

void LogicalGraph::BuildFwStruct() {
  HashMap<std::string, std::vector<LogicalNode*>> op_name2nodes;
  NaiveBuildFwStruct(&op_name2nodes);
  FixSharedModelNodes(op_name2nodes);
  LinkUnpackFw2PackFw(op_name2nodes);
  total_mbn_num_ = 0;
  ForEachNode([&](LogicalNode* node) {
    total_mbn_num_ +=
        node->SoleOp()->model_bns().size() + node->SoleOp()->forward_model_bns().size();
  });
}

void LogicalGraph::NaiveBuildFwStruct(
    HashMap<std::string, std::vector<LogicalNode*>>* op_name2nodes) {
  const DLNetConf& dlnet_conf = Global<JobDesc>::Get()->dlnet_conf();
  const Placement& placement = Global<JobDesc>::Get()->placement();
  HashMap<std::string, std::shared_ptr<ParallelDesc>> name2parallel_desc;
  for (const PlacementGroup& p_group : placement.placement_group()) {
    for (const std::string& op_name : p_group.op_set().op_name()) {
      CHECK(name2parallel_desc
                .emplace(op_name, std::make_shared<ParallelDesc>(p_group.parallel_conf()))
                .second);
    }
  }

  HashMap<LogicalBlobId, std::string> lbi2obn;
  HashMap<LogicalBlobId, LogicalNode*> lbi2producer;
  for (OperatorConf cur_op_conf : dlnet_conf.op()) {
    auto parallel_desc_ptr_it = name2parallel_desc.find(cur_op_conf.name());
    CHECK(parallel_desc_ptr_it != name2parallel_desc.end());
    const std::shared_ptr<ParallelDesc>& parallel_desc_ptr = parallel_desc_ptr_it->second;
    cur_op_conf.set_device_type(parallel_desc_ptr->device_type());
    std::shared_ptr<Operator> cur_op = ConstructOp(cur_op_conf);
    LogicalNode* cur_node = cur_op->NewProperLogicalNode();
    if (cur_node->TypeName() == "PackForward" || cur_node->TypeName() == "UnpackForward") {
      CHECK_EQ(
          0, Global<JobDesc>::Get()->other_conf().piece_size() % parallel_desc_ptr->parallel_num());
    }
    AddAllocatedNode(cur_node);
    cur_node->mut_op_vec() = {cur_op};
    cur_node->SoleOp()->FixParallelDesc(parallel_desc_ptr.get());
    cur_node->mut_parallel_desc() = parallel_desc_ptr;
    for (const std::string& obn : cur_node->SoleOp()->output_bns()) {
      const LogicalBlobId& lbi = cur_node->SoleOp()->BnInOp2Lbi(obn);
      CHECK(lbi2producer.emplace(lbi, cur_node).second);
      CHECK(lbi2obn.emplace(lbi, obn).second);
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
      UpdateEdge2Ibn(edge, ibn);
      UpdateEdge2Obn(edge, lbi2obn.at(lbi));
      Connect(pred_node, edge, cur_node);
    }
  });
}

void LogicalGraph::FixSharedModelNodes(
    const HashMap<std::string, std::vector<LogicalNode*>>& op_name2nodes) {
  HashSet<std::string> all_shared_model_op_names;
  const DLNetConf& dlnet_conf = Global<JobDesc>::Get()->dlnet_conf();
  for (const OpNameSet& op_name_set : dlnet_conf.shared_model_group()) {
    std::vector<std::string> shared_model_op_names(op_name_set.op_name().begin(),
                                                   op_name_set.op_name().end());
    SortAndRemoveDuplication(&shared_model_op_names);
    CHECK_GE(shared_model_op_names.size(), 2);

    auto shared_model_nodes = std::make_shared<std::vector<LogicalNode*>>();
    shared_model_nodes->reserve(shared_model_op_names.size());
    for (const std::string& op_name : shared_model_op_names) {
      CHECK(all_shared_model_op_names.insert(op_name).second);
      CHECK_EQ(op_name2nodes.at(op_name).size(), 1);
      shared_model_nodes->push_back(op_name2nodes.at(op_name).front());
    }

    for (LogicalNode* cur_node : *shared_model_nodes) {
      cur_node->mut_shared_model_nodes() = shared_model_nodes;
    }

    const std::string& shared_op_name = shared_model_nodes->front()->SoleOp()->op_name();
    const ParallelDesc* shared_parallel_desc = shared_model_nodes->front()->parallel_desc().get();
    FOR_RANGE(size_t, i, 1, shared_model_nodes->size()) {
      shared_model_nodes->at(i)->SoleOp()->FixLbiWhenShareModel(shared_op_name);
      CHECK(shared_model_nodes->at(i)->parallel_desc()->Equal(shared_parallel_desc));
    }
  }
}

void LogicalGraph::LinkUnpackFw2PackFw(
    const HashMap<std::string, std::vector<LogicalNode*>>& op_name2nodes) {
  ForEachLogicalNode<PackForwardLogicalNode>([&](PackForwardLogicalNode* pack_fw) {
    const std::string& unpack_name = pack_fw->SoleOp()->op_conf().pack_conf().related_unpack();
    auto it = op_name2nodes.find(unpack_name);
    CHECK(it != op_name2nodes.end());
    CHECK_EQ(1, it->second.size());
    UnpackForwardLogicalNode* unpack_fw =
        dynamic_cast<UnpackForwardLogicalNode*>(it->second.front());
    CHECK(unpack_fw);
    pack_fw->set_related_unpack(unpack_fw);
  });
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

void LogicalGraph::BuildBwStruct() {
  NaiveBuildBwStruct();
  AddBackwardClone();
}

void LogicalGraph::NaiveBuildBwStruct() {
  auto HasActualOutDiff = MakePredicatorHasActualOutDiff(this);
  HashSet<LogicalNode*> nodes_need_bw;
  TopoForEachNode([&](LogicalNode* logical_node) {
    auto fw_node = dynamic_cast<ForwardLogicalNode*>(logical_node);
    if (fw_node == nullptr) { return; }
    if (fw_node->HasOpWithModelBlob()) {
      CHECK(nodes_need_bw.insert(fw_node).second);
      return;
    }
    for (LogicalEdge* edge : fw_node->in_edges()) {
      if (nodes_need_bw.find(edge->src_node()) != nodes_need_bw.end()
          && HasActualOutDiff(fw_node)) {
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
      UpdateEdge2Ibn(bw_edge, edge2ibn_.at(fw_edge));
      UpdateEdge2Obn(bw_edge, edge2obn_.at(fw_edge));
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

void LogicalGraph::AddBackwardClone() {
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
  for (const BackwardCloneInfo& clone_info : clone_infos) { AddOneBackwardClone(clone_info); }
}

void LogicalGraph::AddOneBackwardClone(const BackwardCloneInfo& clone_info) {
  OperatorConf clone_op_conf;
  clone_op_conf.set_name("bw_clone_" + NewUniqueId());
  clone_op_conf.set_device_type(clone_info.succ_node->SoleOp()->device_type());
  clone_op_conf.mutable_clone_conf()->set_out_num(clone_info.edges.size());
  std::shared_ptr<Operator> clone_op = ConstructOp(clone_op_conf);
  LogicalNode* clone_node = NewNode<NormalBackwardLogicalNode>();
  clone_node->mut_op_vec() = {clone_op};
  clone_node->mut_parallel_desc() = clone_info.succ_node->parallel_desc();

  *(clone_op->MutBnInOp2Lbi(clone_op->SoleIbn())) = clone_info.lbi;
  *(clone_op->MutBnInOp2Lbi(clone_op->SoleIdbn())) = clone_info.lbi;
  LogicalEdge* clone_in_diff_edge = NewEdge();
  clone_in_diff_edge->mut_lbis() = {clone_info.lbi};
  Connect(clone_node, clone_in_diff_edge, clone_info.succ_node);
  UpdateEdge2Ibn(clone_in_diff_edge, clone_op->SoleIbn());
  UpdateEdge2Obn(clone_in_diff_edge, edge2obn_.at(clone_info.edges.at(0)));

  FOR_RANGE(size_t, i, 0, clone_info.edges.size()) {
    LogicalBlobId lbi_clone_i = clone_info.lbi;
    lbi_clone_i.set_clone_id(i);
    std::string odbn = clone_op->output_diff_bns().Get(i);
    *(clone_op->MutBnInOp2Lbi(odbn)) = lbi_clone_i;

    LogicalEdge* edge = clone_info.edges.at(i);
    std::vector<LogicalBlobId>& edge_lbis = edge->mut_lbis();
    CHECK_EQ(1, edge_lbis.size());
    edge_lbis.front() = lbi_clone_i;
    LogicalNode* src_node = edge->src_node();
    *(src_node->SoleOp()->MutBnInOp2Lbi(GenDiffBn(edge2ibn_.at(edge)))) = lbi_clone_i;
    DisConnect(edge);
    Connect(src_node, edge, clone_node);
    UpdateEdge2Obn(edge, GenUnDiffBn(odbn));
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
    reduce_sum_conf->add_axis(0);
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
    loss_print_op_conf.set_name(LossPrintPrefix + loss_op->op_name());
    loss_print_op_conf.set_device_type(DeviceType::kCPU);

    auto* loss_print_conf = loss_print_op_conf.mutable_loss_print_conf();
    *(loss_print_conf->mutable_loss_lbi()) = reduce_loss_op->BnInOp2Lbi("out");
    *(loss_print_conf->mutable_loss_instance_num_lbi()->mutable_op_name()) = loss_op->op_name();
    *(loss_print_conf->mutable_loss_instance_num_lbi()->mutable_blob_name()) = "loss_instance_num";
    if (!loss_op->GetValFromCustomizedConf<std::string>("weight").empty()) {
      *(loss_print_conf->mutable_reduction_lbi()) = loss_op->BnInOp2Lbi("reduction_coefficient");
    }
    loss_print_conf->set_weight_scalar(loss_op->GetValFromCustomizedConf<float>("weight_scalar"));
    loss_print_conf->set_reduction_type(
        static_cast<LossReductionType>(loss_op->GetEnumFromCustomizedConf("reduction")));

    std::shared_ptr<Operator> loss_print_op = ConstructOp(loss_print_op_conf);
    ParallelConf loss_print_pr_conf;
    loss_print_pr_conf.set_policy(kDataParallel);
    loss_print_pr_conf.add_device_name("0:cpu:0");
    LossPrintLogicalNode* loss_print_logical = NewNode<LossPrintLogicalNode>();
    loss_print_logical->mut_op_vec() = {loss_print_op};
    loss_print_logical->mut_parallel_desc().reset(new ParallelDesc(loss_print_pr_conf));
    Connect<LogicalNode>(loss_acc_logical, NewEdge(), loss_print_logical);
  });
}

void LogicalGraph::BuildAccuracyPrintStruct() {
  ForEachLogicalNode<AccuracyLogicalNode>([&](AccuracyLogicalNode* accuracy_logical) {
    std::shared_ptr<const Operator> accuracy_op = accuracy_logical->SoleOp();
    // Accuracy Accumulate Logical
    OperatorConf accuracy_acc_op_conf;
    accuracy_acc_op_conf.set_name("accuracy_acc_" + accuracy_op->op_name());
    accuracy_acc_op_conf.set_device_type(accuracy_op->device_type());
    accuracy_acc_op_conf.mutable_accumulate_conf();
    std::shared_ptr<Operator> accuracy_acc_op = ConstructOp(accuracy_acc_op_conf);
    AccuracyAccLogicalNode* accuracy_acc_logical = NewNode<AccuracyAccLogicalNode>();
    accuracy_acc_logical->mut_op_vec() = {accuracy_acc_op};
    accuracy_acc_logical->mut_parallel_desc() = accuracy_logical->parallel_desc();
    Connect<LogicalNode>(accuracy_logical, NewEdge(), accuracy_acc_logical);
    // Accuracy Print Logical
    OperatorConf accuracy_print_op_conf;
    accuracy_print_op_conf.set_name(AccuracyPrintPrefix + accuracy_op->op_name());
    accuracy_print_op_conf.set_device_type(DeviceType::kCPU);

    auto* accuracy_print_conf = accuracy_print_op_conf.mutable_accuracy_print_conf();
    accuracy_print_conf->set_top_k_print(accuracy_op->op_conf().accuracy_conf().top_k());
    *(accuracy_print_conf->mutable_accuracy_lbi()) = accuracy_op->BnInOp2Lbi("accuracy");
    *(accuracy_print_conf->mutable_accuracy_instance_num_lbi()->mutable_op_name()) =
        accuracy_op->op_name();
    *(accuracy_print_conf->mutable_accuracy_instance_num_lbi()->mutable_blob_name()) =
        "accuracy_instance_num";

    std::shared_ptr<Operator> accuracy_print_op = ConstructOp(accuracy_print_op_conf);
    ParallelConf accuracy_print_pr_conf;
    accuracy_print_pr_conf.set_policy(kDataParallel);
    accuracy_print_pr_conf.add_device_name("0:cpu:0");
    AccuracyPrintLogicalNode* accuracy_print_logical = NewNode<AccuracyPrintLogicalNode>();
    accuracy_print_logical->mut_op_vec() = {accuracy_print_op};
    accuracy_print_logical->mut_parallel_desc().reset(new ParallelDesc(accuracy_print_pr_conf));
    Connect<LogicalNode>(accuracy_acc_logical, NewEdge(), accuracy_print_logical);
  });
}

bool LogicalGraph::MustHaveModelDiffAcc() {
  bool must_have_model_diff_acc = false;
  ForEachLogicalNode<ForwardLogicalNode>(
      [&must_have_model_diff_acc](ForwardLogicalNode* fw_logical) {
        if (must_have_model_diff_acc) { return; }
        if (fw_logical->TypeName() == "PackForward" || fw_logical->TypeName() == "UnpackForward"
            || fw_logical->TypeName() == "RepeatForward") {
          must_have_model_diff_acc = true;
          return;
        }
      });
  return must_have_model_diff_acc;
}

void LogicalGraph::BuildModelStruct(bool is_train) {
  HashMap<const LogicalNode*, NormalMdUpdtLogicalNode*> first_shared2mdupdt;
  HashMap<const LogicalNode*, ReduceCtx> fw_node2reduce_ctx;
  bool must_have_model_diff_acc = MustHaveModelDiffAcc();
  ForEachLogicalNode<ForwardLogicalNode>([&](ForwardLogicalNode* fw_logical) {
    if (Global<JobDesc>::Get()->enable_write_snapshot() && fw_logical->HasOpWithForwardModelBlob()
        && fw_logical->SoleOp()->op_conf().trainable()) {
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
      BackwardLogicalNode* bw_logical = fw_logical->bw_node();
      if (bw_logical) { Connect<LogicalNode>(md_updt_logical, NewEdge(), bw_logical); }
      // Model Diff Accumulate Logical
      if (is_train && fw_logical->HasOpWithModelBlob()) {
        CHECK_NOTNULL(bw_logical);
        LogicalNode* md_diff_acc_logical = nullptr;
        if (must_have_model_diff_acc || Global<JobDesc>::Get()->NumOfPiecesInBatch() > 1) {
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
          ReduceCtx reduce_ctx;
          reduce_ctx.fw_logicals.emplace_back(fw_logical);
          reduce_ctx.bw_logicals.emplace_back(bw_logical);
          reduce_ctx.md_diff_acc_logicals.emplace_back(md_diff_acc_logical);
          reduce_ctx.md_updt_logicals.emplace_back(md_updt_logical);
          CHECK(fw_node2reduce_ctx.emplace(fw_logical, reduce_ctx).second);
        } else {
          Connect<LogicalNode>(md_diff_acc_logical, NewEdge(), md_updt_logical);
        }
      }
    }
  });
  for (int i = 0; i < fw_node_groups_.size(); ++i) {
    auto& fw_node_group = fw_node_groups_[i];
    ReduceCtx group_reduce_ctx;
    group_reduce_ctx.order_in_logical_graph = i;
    int order_in_reduce_group = 0;
    for (auto& fw_node : fw_node_group) {
      auto reduce_ctx_it = fw_node2reduce_ctx.find(fw_node);
      if (reduce_ctx_it != fw_node2reduce_ctx.end()) {
        auto& reduce_ctx = reduce_ctx_it->second;
        group_reduce_ctx.fw_logicals.emplace_back(reduce_ctx.fw_logicals.at(0));
        group_reduce_ctx.bw_logicals.emplace_back(reduce_ctx.bw_logicals.at(0));
        group_reduce_ctx.md_diff_acc_logicals.emplace_back(reduce_ctx.md_diff_acc_logicals.at(0));
        group_reduce_ctx.md_updt_logicals.emplace_back(reduce_ctx.md_updt_logicals.at(0));
        auto* md_updt = dynamic_cast<NormalMdUpdtLogicalNode*>(reduce_ctx.md_updt_logicals.at(0));
        md_updt->set_order_in_reduce_group(order_in_reduce_group++);
      }
    }
    if (group_reduce_ctx.fw_logicals.size() > 0) { BuildReduceStruct(group_reduce_ctx); }
  }
  SetupNormalMdUpdtOp();
}

void LogicalGraph::BuildReduceStruct(const ReduceCtx& reduce_ctx) {
  CHECK_GT(reduce_ctx.fw_logicals.size(), 0);
  std::shared_ptr<const ParallelDesc> src_pd = reduce_ctx.fw_logicals[0]->parallel_desc();

  OperatorConf reduce_concat_op_conf;
  reduce_concat_op_conf.set_name("reduce_concat_" + NewUniqueId());
  reduce_concat_op_conf.set_device_type(src_pd->device_type());
  reduce_concat_op_conf.mutable_reduce_concat_conf()->set_in_num(reduce_ctx.fw_logicals.size());
  LogicalNode* reduce_concat_node = NewNode<ReduceConcatLogicalNode>();
  reduce_concat_node->mut_op_vec() = {ConstructOp(reduce_concat_op_conf)};
  reduce_concat_node->mut_parallel_desc() = src_pd;

  // We can not add ctrl edges between all_reduce nodes due to the implementation of nccl.
  // So we add ctrl edges between ReduceIdentityTaskNodes which are followed by
  // all_reduce nodes;
  OperatorConf reduce_identity_conf;
  reduce_identity_conf.set_name("reduce_identity_" + NewUniqueId());
  reduce_identity_conf.set_device_type(src_pd->device_type());
  reduce_identity_conf.mutable_reduce_identity_conf();
  auto* reduce_identity_node = NewNode<ReduceIdentityLogicalNode>();
  reduce_identity_node->mut_op_vec() = {ConstructOp(reduce_identity_conf)};
  reduce_identity_node->mut_parallel_desc() = src_pd;
  reduce_identity_node->set_order_in_logical_graph(reduce_ctx.order_in_logical_graph);

  OperatorConf reduce_split_op_conf;
  reduce_split_op_conf.set_name("reduce_split_" + NewUniqueId());
  reduce_split_op_conf.set_device_type(src_pd->device_type());
  reduce_split_op_conf.mutable_reduce_split_conf()->set_out_num(reduce_ctx.fw_logicals.size());
  auto* reduce_split_node = NewNode<ReduceSplitLogicalNode>();
  reduce_split_node->mut_op_vec() = {ConstructOp(reduce_split_op_conf)};
  reduce_split_node->mut_parallel_desc() = src_pd;
  reduce_split_node->set_order_in_logical_graph(reduce_ctx.order_in_logical_graph);

  for (auto& md_diff_acc_node : reduce_ctx.md_diff_acc_logicals) {
    Connect(md_diff_acc_node, NewEdge(), reduce_concat_node);
  }
  Connect(reduce_concat_node, NewEdge(), static_cast<LogicalNode*>(reduce_identity_node));
  AddAllReduce(reduce_identity_node, reduce_split_node);
  for (auto& md_updt_node : reduce_ctx.md_updt_logicals) {
    Connect(static_cast<LogicalNode*>(reduce_split_node), NewEdge(), md_updt_node);
  }
}

void LogicalGraph::AddAllReduce(LogicalNode* src, LogicalNode* dst) {
  std::shared_ptr<const ParallelDesc> src_pd = src->parallel_desc();
  std::shared_ptr<const ParallelDesc> dst_pd = dst->parallel_desc();
  CHECK_EQ(src_pd->parallel_num(), dst_pd->parallel_num());
  CHECK_EQ(src_pd->device_type(), dst_pd->device_type());

  if (Global<JobDesc>::Get()->enable_nccl() && src_pd->device_type() == DeviceType::kGPU) {
    if (src_pd->sorted_machine_ids().size() == 1
        || Global<JobDesc>::Get()->use_nccl_inter_node_communication()) {
      AddNcclAllReduce(src, dst);
    } else if (src_pd->device_num_of_each_machine() == 1) {
      AddReduceScatterAddGatherNodes(src, dst, ReduceRankCtx());
    } else {
      AddNcclReduceScatterAndAllGather(src, dst);
    }
  } else {
    AddReduceScatterAddGatherNodes(src, dst, ReduceRankCtx());
  }
}

void LogicalGraph::AddNcclReduceScatterAndAllGather(LogicalNode* src, LogicalNode* dst) {
  std::shared_ptr<const ParallelDesc> src_pd = src->parallel_desc();

  ReduceRankCtx rank_ctx = ReduceRankCtx().CtxWithScatter(src_pd->device_num_of_each_machine());

  OperatorConf nccl_reduce_scatter_op_conf;
  nccl_reduce_scatter_op_conf.set_name("nccl_reduce_scatter_" + NewUniqueId());
  nccl_reduce_scatter_op_conf.set_device_type(src_pd->device_type());
  nccl_reduce_scatter_op_conf.mutable_nccl_reduce_scatter_conf();
  NcclReduceScatterLogicalNode* nccl_reduce_scatter_node = NewNode<NcclReduceScatterLogicalNode>();
  nccl_reduce_scatter_node->mut_op_vec() = {ConstructOp(nccl_reduce_scatter_op_conf)};
  nccl_reduce_scatter_node->mut_parallel_desc() = src_pd;
  nccl_reduce_scatter_node->mut_rank_ctx() = rank_ctx;
  Connect<LogicalNode>(src, NewEdge(), nccl_reduce_scatter_node);

  OperatorConf nccl_all_gather_op_conf;
  nccl_all_gather_op_conf.set_name("nccl_all_gather_" + NewUniqueId());
  nccl_all_gather_op_conf.set_device_type(src_pd->device_type());
  nccl_all_gather_op_conf.mutable_nccl_all_gather_conf();
  NcclAllGatherLogicalNode* nccl_all_gather_node = NewNode<NcclAllGatherLogicalNode>();
  nccl_all_gather_node->mut_op_vec() = {ConstructOp(nccl_all_gather_op_conf)};
  nccl_all_gather_node->mut_parallel_desc() = src_pd;
  nccl_all_gather_node->mut_rank_ctx() = rank_ctx;
  Connect<LogicalNode>(nccl_all_gather_node, NewEdge(), dst);

  AddReduceScatterAddGatherNodes(nccl_reduce_scatter_node, nccl_all_gather_node, rank_ctx);
}

void LogicalGraph::AddNcclAllReduce(LogicalNode* src, LogicalNode* dst) {
  std::shared_ptr<const ParallelDesc> src_pd = src->parallel_desc();
  OperatorConf nccl_all_reduce_op_conf;
  nccl_all_reduce_op_conf.set_name("nccl_all_reduce_" + NewUniqueId());
  nccl_all_reduce_op_conf.set_device_type(src_pd->device_type());
  nccl_all_reduce_op_conf.mutable_nccl_all_reduce_conf();
  NcclAllReduceLogicalNode* nccl_all_reduce_node = NewNode<NcclAllReduceLogicalNode>();
  nccl_all_reduce_node->mut_op_vec() = {ConstructOp(nccl_all_reduce_op_conf)};
  nccl_all_reduce_node->mut_parallel_desc() = src_pd;
  nccl_all_reduce_node->mut_rank_ctx() = ReduceRankCtx().CtxWithScatter(src_pd->parallel_num());
  Connect<LogicalNode>(src, NewEdge(), nccl_all_reduce_node);
  Connect<LogicalNode>(nccl_all_reduce_node, NewEdge(), dst);
}

void LogicalGraph::AddReduceScatterAddGatherNodes(LogicalNode* src, LogicalNode* dst,
                                                  const ReduceRankCtx& prev_rank_ctx) {
  std::shared_ptr<const ParallelDesc> src_pd = src->parallel_desc();

  int64_t segment_count =
      prev_rank_ctx.TotalSegmentCount() == 1
          ? (src_pd->device_num_of_each_machine() == 1 ? src_pd->sorted_machine_ids().size()
                                                       : src_pd->device_num_of_each_machine())
          : src_pd->sorted_machine_ids().size();

  ReduceRankCtx current_rank_ctx = prev_rank_ctx.CtxWithScatter(segment_count);
  ReduceScatterLogicalNode* reduce_scatter_node = NewNode<ReduceScatterLogicalNode>();
  reduce_scatter_node->mut_parallel_desc() = src_pd;
  reduce_scatter_node->mut_rank_ctx() = current_rank_ctx;
  Connect<LogicalNode>(src, NewEdge(), reduce_scatter_node);

  ReduceAddLogicalNode* reduce_add_node = NewNode<ReduceAddLogicalNode>();
  reduce_add_node->mut_parallel_desc() = src_pd;
  reduce_add_node->mut_rank_ctx() = current_rank_ctx;
  Connect<LogicalNode>(reduce_scatter_node, NewEdge(), reduce_add_node);

  ReduceGatherLogicalNode* reduce_gather_node = NewNode<ReduceGatherLogicalNode>();
  reduce_gather_node->mut_parallel_desc() = src_pd;
  reduce_gather_node->mut_rank_ctx() = current_rank_ctx;

  if (current_rank_ctx.TotalSegmentCount() == src_pd->parallel_num()) {
    Connect<LogicalNode>(reduce_add_node, NewEdge(), reduce_gather_node);
  } else {
    AddReduceScatterAddGatherNodes(reduce_add_node, reduce_gather_node, current_rank_ctx);
  }

  Connect<LogicalNode>(reduce_gather_node, NewEdge(), dst);
}

void LogicalGraph::SetupNormalMdUpdtOp() {
  ForEachLogicalNode<NormalMdUpdtLogicalNode>([](NormalMdUpdtLogicalNode* node) {
    if (node->in_edges().size() < 1) { return; }
    // Add shared_model_diff_add_op
    OperatorConf op_conf;
    op_conf.set_name("md_diff_add_" + NewUniqueId());
    op_conf.set_device_type(node->parallel_desc()->device_type());
    op_conf.mutable_shared_model_diff_add_conf()->set_in_num(node->in_edges().size());
    node->mut_op_vec() = {ConstructOp(op_conf)};
  });
}

MdSaveLogicalNode* LogicalGraph::BuildMdSaveStruct(const ForwardLogicalNode* fw_logical,
                                                   LogicalNode* need_save_logical) {
  OperatorConf md_save_op_conf;
  md_save_op_conf.set_name("md_save_" + NewUniqueId());
  md_save_op_conf.set_device_type(DeviceType::kCPU);
  md_save_op_conf.mutable_model_save_conf();
  auto model_save_op = ConstructOp(md_save_op_conf);
  auto md_save_logical = NewNode<MdSaveLogicalNode>();
  md_save_logical->mut_op_vec() = {model_save_op};
  auto md_save_pr_desc = new ParallelDesc(*(fw_logical->parallel_desc()));
  md_save_pr_desc->set_device_type(DeviceType::kCPU);
  if (fw_logical->parallel_desc()->policy() == ParallelPolicy::kDataParallel) {
    md_save_pr_desc->RandomSelectOneDeviceAndRemoveTheOthers();
  }
  if (Global<JobDesc>::Get()->write_snapshot_to_master()) {
    md_save_pr_desc->UseCPUDevicesOnMaster();
  }
  md_save_logical->mut_parallel_desc().reset(md_save_pr_desc);
  Connect<LogicalNode>(need_save_logical, NewEdge(), md_save_logical);
  return md_save_logical;
}

NormalMdUpdtLogicalNode* LogicalGraph::BuildNormalMdUpdtAndMdSaveStruct(
    bool is_train, ForwardLogicalNode* fw_logical) {
  NormalMdUpdtLogicalNode* md_updt_logical = NewNode<NormalMdUpdtLogicalNode>();
  md_updt_logical->mut_parallel_desc() = fw_logical->parallel_desc();
  if (Global<JobDesc>::Get()->enable_write_snapshot()) {
    // for model
    BuildMdSaveStruct(fw_logical, md_updt_logical);
    // TODO: remove the following ugly hard coded `if'
    if (Global<JobDesc>::Get()->other_conf().train_conf().model_update_conf().has_momentum_conf()
        || Global<JobDesc>::Get()->other_conf().train_conf().model_update_conf().has_adam_conf()) {
      // for forward_model
      BuildMdSaveStruct(fw_logical, md_updt_logical);
    }
  }
  return md_updt_logical;
}

void LogicalGraph::ConnectFwToBw() {
  ForEachLogicalNode<BackwardLogicalNode>([this](BackwardLogicalNode* bw_node) {
    if (bw_node->fw_node() == nullptr) { return; }
    Connect<LogicalNode>(bw_node->fw_node(), NewEdge(), bw_node);
  });
}

void LogicalGraph::UpdateEdge2Ibn(const LogicalEdge* edge, const std::string& ibn) {
  if (!ibn.empty()) { edge2ibn_[edge] = ibn; }
}

void LogicalGraph::UpdateEdge2Obn(const LogicalEdge* edge, const std::string& obn) {
  if (!obn.empty()) { edge2obn_[edge] = obn; }
}

}  // namespace oneflow
