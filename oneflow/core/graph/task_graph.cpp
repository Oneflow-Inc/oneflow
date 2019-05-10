#include "oneflow/core/graph/task_graph.h"
#include "oneflow/core/graph/normal_forward_compute_task_node.h"
#include "oneflow/core/graph/normal_backward_compute_task_node.h"
#include "oneflow/core/graph/normal_model_update_compute_task_node.h"
#include "oneflow/core/graph/chain_graph.h"
#include "oneflow/core/graph/boxing_task_node.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/graph/reduce_add_compute_task_node.h"
#include "oneflow/core/graph/reduce_gather_compute_task_node.h"
#include "oneflow/core/graph/reduce_split_compute_task_node.h"
#include "oneflow/core/register/runtime_blob_desc.h"
#include "oneflow/core/job/thrd_id_generator.h"
#include "oneflow/core/graph/reduce_identity_task_node.h"
#include "oneflow/core/operator/variable_op.h"
#include "oneflow/core/operator/constant_op.h"

namespace oneflow {

namespace {

bool IsConnectToTickOp(const TaskNode* node) {
  const auto* comp_task_node = dynamic_cast<const CompTaskNode*>(node);
  if (comp_task_node == nullptr) { return false; }
  if (comp_task_node->logical_node()->op_vec().size() != 1) { return false; }
  const Operator* op = comp_task_node->logical_node()->SoleOp().get();
  if (dynamic_cast<const VariableOp*>(op) != nullptr) { return true; }
  if (dynamic_cast<const ConstantOp*>(op) != nullptr) { return true; }
  return false;
}

template<typename ReduceType = ReduceIdentityLogicalNode>
typename std::enable_if<std::is_same<ReduceType, ReduceIdentityLogicalNode>::value
                            || std::is_same<ReduceType, ReduceSplitLogicalNode>::value,
                        std::function<int32_t(const LogicalNode*)>>::type
MakeGetterReduceTaskNodeCtrlOrder(const LogicalGraph& logical_graph) {
  std::vector<const ReduceType*> logical_nodes;
  logical_graph.ForEachNode([&](LogicalNode* node) {
    auto* logical_node = dynamic_cast<ReduceType*>(node);
    if (logical_node == nullptr) { return; }
    logical_nodes.push_back(logical_node);
  });
  std::sort(logical_nodes.begin(), logical_nodes.end(),
            [](const ReduceType* lhs, const ReduceType* rhs) {
              return lhs->order_in_logical_graph() < rhs->order_in_logical_graph();
            });
  auto logical_node2ctrl_order = std::make_shared<HashMap<const LogicalNode*, int32_t>>();
  int32_t lazy_count = Global<JobDesc>::Get()->all_reduce_lazy_ratio() * logical_nodes.size();
  for (int32_t i = 0; i < logical_nodes.size(); ++i) {
    int32_t ctrl_order = 0;
    if (i > lazy_count) {
      ctrl_order = -i;
    } else {
      ctrl_order = i;
    }
    (*logical_node2ctrl_order)[logical_nodes[i]] = ctrl_order;
  }
  return [logical_node2ctrl_order](const LogicalNode* identity_node) {
    return logical_node2ctrl_order->at(identity_node);
  };
}

void ForEachDeviceSrcUntrainableNode(const std::vector<NormalForwardCompTaskNode*>& fw_nodes,
                                     const std::function<void(CompTaskNode*)>& Handler) {
  HashSet<const TaskNode*> fw_nodes_set(fw_nodes.begin(), fw_nodes.end());
  auto IsSourceTaskNode = [&](NormalForwardCompTaskNode* node) {
    for (TaskEdge* edge : node->in_edges()) {
      if (fw_nodes_set.find(edge->src_node()) != fw_nodes_set.end()) { return false; }
    }
    return true;
  };
  auto HasBwNode = [&](NormalForwardCompTaskNode* node) {
    const auto* fw_logical_node = dynamic_cast<const ForwardLogicalNode*>(node->logical_node());
    return fw_logical_node->bw_node() != nullptr;
  };
  for (NormalForwardCompTaskNode* fw_node : fw_nodes) {
    if (IsSourceTaskNode(fw_node) && !HasBwNode(fw_node)) { Handler(fw_node); }
  }
}

}  // namespace

TaskGraph::TaskGraph(std::unique_ptr<const LogicalGraph>&& logical_gph) {
  logical_gph_ = std::move(logical_gph);
  HashMap<const LogicalNode*, std::vector<CompTaskNode*>> logical2sorted_comp_tasks;
  HashMap<const LogicalNode*, std::vector<TaskNode*>> logical2sorted_in_box;
  HashMap<const LogicalNode*, std::vector<TaskNode*>> logical2sorted_out_box;
  HashMap<CompTaskNode*, HashMap<int64_t, std::vector<TaskNode*>>> buf_task;
  const JobDesc* job_desc = Global<JobDesc>::Get();
  auto MutBufTask = [&](CompTaskNode* task_node, int64_t machine_id, int32_t mem_zone_id) {
    auto& buf_vec = buf_task[task_node][machine_id];
    if (buf_vec.empty()) { buf_vec.assign(job_desc->MemZoneNum(), nullptr); }
    return &(buf_vec.at(mem_zone_id));
  };

  std::vector<int64_t> cpu_device_offset(job_desc->TotalMachineNum(), 0);
  auto AllocateCpuThrdIdEvenly = [&](const TaskNode* task_node) {
    CHECK(!task_node->IsPersistence());
    int64_t ret = -1;
    int64_t& offset = cpu_device_offset.at(task_node->machine_id());
    ret = Global<IDMgr>::Get()->GetCpuDeviceThrdId(offset);
    offset = (offset + 1) % job_desc->CpuDeviceNum();
    return ret;
  };

  std::vector<std::pair<int64_t, CompTaskNode*>> machine_persistence_task_vec;
  logical_gph_->ForEachNode([&](const LogicalNode* logical_node) {
    logical_node->GenSortedCompTaskNodes(
        AllocateCpuThrdIdEvenly, &machine_persistence_task_vec, [&](CompTaskNode* comp_task_node) {
          AddAllocatedNode(comp_task_node);
          logical2sorted_comp_tasks[logical_node].push_back(comp_task_node);
          comp_task_node->set_area_id(logical_node->GetAreaId());
        });
  });

  logical_gph_->ForEachNode([&](LogicalNode* logical) {
    PackForwardLogicalNode* pack_fw = dynamic_cast<PackForwardLogicalNode*>(logical);
    if (pack_fw == nullptr) { return; }
    const UnpackForwardLogicalNode* unpack_fw = pack_fw->related_unpack();
    const std::vector<CompTaskNode*>& pack_fw_tasks = logical2sorted_comp_tasks.at(pack_fw);
    const std::vector<CompTaskNode*>& unpack_fw_tasks = logical2sorted_comp_tasks.at(unpack_fw);
    CHECK_EQ(pack_fw_tasks.size(), unpack_fw_tasks.size());
    for (size_t i = 0; i < pack_fw_tasks.size(); ++i) {
      dynamic_cast<PackForwardCompTaskNode*>(pack_fw_tasks.at(i))
          ->set_related_unpack(dynamic_cast<UnpackForwardCompTaskNode*>(unpack_fw_tasks.at(i)));
    }
  });

  GeneratePersistenceThrdId(machine_persistence_task_vec);
  logical_gph_->ForEachEdge([&](const LogicalEdge* logical_edge) {
    BldSubTskGphMthd method =
        GetMthdForBldSubTskGph(logical_edge->src_node(), logical_edge->dst_node());
    (this->*method)(logical_edge->src_node(), logical_edge->dst_node(),
                    logical2sorted_comp_tasks.at(logical_edge->src_node()),
                    logical2sorted_comp_tasks.at(logical_edge->dst_node()), &logical2sorted_in_box,
                    &logical2sorted_out_box, MutBufTask, AllocateCpuThrdIdEvenly);
    SetAreaIdForNewNodes(logical_edge->src_node(), logical_edge->dst_node());
  });
  logical_gph_->ForEachNecessaryCtrlEdge(
      [&](const LogicalNode* src, const LogicalNode* dst, int64_t ctrl_regst_num) {
        const auto& src_task_nodes = logical2sorted_comp_tasks.at(src);
        const auto& dst_task_nodes = logical2sorted_comp_tasks.at(dst);
        ConnectCtrlEdges(src_task_nodes, dst_task_nodes, ctrl_regst_num);
      });

  MergeChainAndSetOrderInGraphForEachNode();
  ToDotWithAutoFilePath();
}

void TaskGraph::ConnectCtrlEdges(const std::vector<CompTaskNode*>& src_task_nodes,
                                 const std::vector<CompTaskNode*>& dst_task_nodes,
                                 int64_t ctrl_regst_num) {
  CHECK_EQ(src_task_nodes.size(), dst_task_nodes.size());
  FOR_RANGE(int32_t, i, 0, src_task_nodes.size()) {
    std::string regst_desc_name;
    RegstDesc* ctrl_regst_desc =
        src_task_nodes.at(i)->BuildCtrlRegstDesc(dst_task_nodes.at(i), &regst_desc_name);
    ctrl_regst_desc->UpdtMinRegstNumIfNeed(ctrl_regst_num);
    ctrl_regst_desc->UpdtMaxRegstNumIfNeed(ctrl_regst_num);
    ctrl_regst_desc->mut_regst_desc_type()->mutable_ctrl_regst_desc()->set_returned_regst_num(
        ctrl_regst_num);

    TaskEdge* edge = NewEdge();
    Connect<TaskNode>(src_task_nodes.at(i), edge, dst_task_nodes.at(i));
    src_task_nodes.at(i)->BindEdgeWithProducedRegst(edge, regst_desc_name);
  }
}

void TaskGraph::GeneratePersistenceThrdId(
    const std::vector<std::pair<int64_t, CompTaskNode*>>& persistence_nodes) {
  std::vector<std::pair<int64_t, TaskType>> machine_task_type_vec;
  for (auto pair : persistence_nodes) {
    machine_task_type_vec.emplace_back(std::make_pair(pair.first, pair.second->GetTaskType()));
  }

  ThrdIdGenerator generator(machine_task_type_vec, Global<IDMgr>::Get()->BasePersistenceThrdId());
  for (const auto pair : persistence_nodes) {
    int64_t thrd_id = generator.GenerateThrdId(pair.first, pair.second->GetTaskType());
    pair.second->set_thrd_id(thrd_id);
  }
}

void TaskGraph::MdUpdtDelayedTopoForEachNode(std::function<void(TaskNode* node)> Handler) const {
  HashSet<const TaskNode*> built_nodes;
  auto Build = [&](TaskNode* node) {
    CHECK(built_nodes.emplace(node).second);
    Handler(node);
  };
  AcyclicTopoForEachNode([](TaskNode* node) { return node->GetTaskType() != kNormalMdUpdt; },
                         Build);
  AcyclicTopoForEachNode([](TaskNode* node) { return node->GetTaskType() == kNormalMdUpdt; },
                         Build);
  ForEachNode([&](TaskNode* node) { CHECK(built_nodes.find(node) != built_nodes.end()); });
}

void TaskGraph::AcyclicTopoForEachNode(std::function<bool(TaskNode* node)> IsAllowedStartNode,
                                       std::function<void(TaskNode* node)> Handler) const {
  auto ForEachInNode = [&](TaskNode* node, const std::function<void(TaskNode*)>& Handler) {
    node->ForEachNodeOnInEdge([&](TaskNode* node_on_in_edge) {
      if (IsBackEdge(node_on_in_edge, node)) { return; }
      Handler(const_cast<TaskNode*>(node_on_in_edge));
    });
  };
  auto ForEachOutNode = [&](TaskNode* node, const std::function<void(TaskNode*)>& Handler) {
    node->ForEachNodeOnOutEdge([&](TaskNode* node_on_out_edge) {
      if (IsBackEdge(node, node_on_out_edge)) { return; }
      Handler(const_cast<TaskNode*>(node_on_out_edge));
    });
  };
  auto IsSourceNode = [&](TaskNode* node) {
    int32_t in_node_num = 0;
    ForEachInNode(node, [&](TaskNode* in_node) { ++in_node_num; });
    return in_node_num == 0;
  };
  std::list<TaskNode*> starts;
  ForEachNode([&](TaskNode* node) {
    if (IsSourceNode(node) && IsAllowedStartNode(node)) { starts.push_back(node); }
  });
  // DfsTopo will cause inappropriate chain graph
  TopoForEachNode(starts, ForEachInNode, ForEachOutNode, Handler);
}

void TaskGraph::AcyclicTopoForEachNode(std::function<void(TaskNode* node)> Handler) const {
  return AcyclicTopoForEachNode([](TaskNode*) { return true; }, Handler);
}

void TaskGraph::RemoveEmptyRegsts() {
  ForEachNode([&](TaskNode* node) { node->EraseZeroSizeProducedBlob(); });
  ForEachNode([&](TaskNode* node) { node->EraseZeroSizeConsumedRegst(); });
  ForEachNode([&](TaskNode* node) { node->EraseZeroSizeProducedRegst(); });
  ForEachNode([&](TaskNode* node) { node->UnbindBnWithEmptyRegst(); });
}

void TaskGraph::AddOrderingCtrlEdgeInSameChain() { BuildCtrlRegstDescInSameChain(); }

void TaskGraph::MergeChainAndSetOrderInGraphForEachNode() {
  ChainGraph chain_graph(*this);
  const auto& ordered_chain_nodes = chain_graph.OrderdedChainNodes();
  int64_t order_in_graph = 0;
  for (auto& chain_node : ordered_chain_nodes) {
    auto& ordered_in_chain = chain_node->TaskNodes();
    int64_t chain_id = chain_node->chain_id();
    for (auto& task_node : ordered_in_chain) {
      task_node->set_chain_id(chain_id);
      task_node->set_order_in_graph(order_in_graph);
      ordered_task_nodes_.emplace_back(task_node);
      ++order_in_graph;
    }
  }
}

void TaskGraph::BuildCtrlRegstDescInSameChain() {
  HashMap<int64_t, TaskNode*> chain_id2node;
  for (auto* node : ordered_task_nodes_) {
    if (IsConnectToTickOp(node)) { continue; }
    int64_t chain_id = node->chain_id();
    auto iter = chain_id2node.find(chain_id);
    if (iter == chain_id2node.end()) {
      CHECK(chain_id2node.emplace(chain_id, node).second);
    } else {
      iter->second->BuildCtrlRegstDescIfNeed(node);
      iter->second = node;
    }
  }
}

void TaskGraph::AddReduceSequenceCtrlEdges() {
  HashMap<int64_t, std::vector<ReduceSplitCompTaskNode*>> global_thrd_id2split_nodes;
  for (auto* node : ordered_task_nodes_) {
    auto* split_node = dynamic_cast<ReduceSplitCompTaskNode*>(node);
    if (split_node == nullptr) { continue; }
    int64_t global_thrd_id = Global<IDMgr>::Get()->GlobalThrdId4TaskId(split_node->task_id());
    global_thrd_id2split_nodes[global_thrd_id].push_back(split_node);
  }
  auto GetCtrlOrder = MakeGetterReduceTaskNodeCtrlOrder<ReduceSplitLogicalNode>(*logical_gph_);
  for (auto& pair : global_thrd_id2split_nodes) {
    auto& split_nodes = pair.second;
    std::sort(split_nodes.begin(), split_nodes.end(),
              [&](ReduceSplitCompTaskNode* lhs, ReduceSplitCompTaskNode* rhs) {
                return GetCtrlOrder(lhs->logical_node()) < GetCtrlOrder(rhs->logical_node());
              });
    ReduceSplitCompTaskNode* prev_split_node = split_nodes.at(0);
    for (auto* split_node : split_nodes) {
      if (prev_split_node != split_node) {
        auto* to_node = split_node->GetPrevReduceTaskNode(TaskType::kReduceIdentity);
        TaskNode* from_node = prev_split_node;
        if (GetCtrlOrder(split_node->logical_node()) < 0) {
          from_node = prev_split_node->GetPrevReduceTaskNode(TaskType::kReduceIdentity);
        }
        from_node->BuildCtrlRegstDescIfNeed(to_node);
      }
      prev_split_node = split_node;
    }
  }
}

void TaskGraph::AddMdUpdtCtrlEdgesWithinReduceSplitNode() {
  auto GetOrderInReduceGroup = [&](NormalMdUpdtCompTaskNode* md_updt_node) {
    const auto* logical_node =
        dynamic_cast<const NormalMdUpdtLogicalNode*>(md_updt_node->logical_node());
    return logical_node->order_in_reduce_group();
  };
  for (auto* node : ordered_task_nodes_) {
    auto* split_node = dynamic_cast<ReduceSplitCompTaskNode*>(node);
    if (split_node == nullptr) { continue; }
    std::vector<NormalMdUpdtCompTaskNode*> md_updt_nodes;
    split_node->ForEachNodeOnOutEdge([&](TaskNode* node) {
      auto* md_updt_node = dynamic_cast<NormalMdUpdtCompTaskNode*>(node);
      if (md_updt_node == nullptr) { return; }
      md_updt_nodes.push_back(md_updt_node);
    });
    std::sort(md_updt_nodes.begin(), md_updt_nodes.end(),
              [&](NormalMdUpdtCompTaskNode* lhs, NormalMdUpdtCompTaskNode* rhs) {
                return GetOrderInReduceGroup(lhs) < GetOrderInReduceGroup(rhs);
              });
    NormalMdUpdtCompTaskNode* prev_md_updt = md_updt_nodes.at(0);
    for (auto* md_updt_node : md_updt_nodes) {
      if (md_updt_node != prev_md_updt) { prev_md_updt->BuildCtrlRegstDescIfNeed(md_updt_node); }
      prev_md_updt = md_updt_node;
    }
  }
}

void TaskGraph::AddReduceNoBwForwardNodeOverlapingCtrlEdges() {
  HashMap<int64_t, std::vector<ReduceIdentityCompTaskNode*>> global_thrd_id2identity_nodes;
  HashMap<std::pair<int64_t, int64_t>, std::vector<NormalForwardCompTaskNode*>>
      global_dev_phy_id2fw_nodes;
  const auto* id_mgr = Global<IDMgr>::Get();
  for (auto* node : ordered_task_nodes_) {
    if (id_mgr->GetDeviceTypeFromThrdId(node->thrd_id()) == DeviceType::kCPU) { continue; }
    int64_t global_thrd_id = id_mgr->GlobalThrdId4TaskId(node->task_id());
    auto* identity_node = dynamic_cast<ReduceIdentityCompTaskNode*>(node);
    auto* fw_node = dynamic_cast<NormalForwardCompTaskNode*>(node);
    if (identity_node != nullptr) {
      global_thrd_id2identity_nodes[global_thrd_id].push_back(identity_node);
    } else if (fw_node != nullptr) {
      int64_t dev_phy_id = id_mgr->GetGpuPhyIdFromThrdId(node->thrd_id());
      global_dev_phy_id2fw_nodes[std::make_pair(node->machine_id(), dev_phy_id)].push_back(fw_node);
    } else {
      // do nothing
    }
  }
  auto GetIdentityNodeOrder = [&](const ReduceIdentityCompTaskNode* id_node) {
    const auto* id_logical_node =
        dynamic_cast<const ReduceIdentityLogicalNode*>(id_node->logical_node());
    return id_logical_node->order_in_logical_graph();
  };
  for (auto& pair : global_thrd_id2identity_nodes) {
    auto& identity_nodes = pair.second;
    std::sort(identity_nodes.begin(), identity_nodes.end(),
              [&](ReduceIdentityCompTaskNode* lhs, ReduceIdentityCompTaskNode* rhs) {
                return GetIdentityNodeOrder(lhs) < GetIdentityNodeOrder(rhs);
              });
    auto* first_identity_node = identity_nodes.at(0);
    int64_t machine_id = first_identity_node->machine_id();
    int64_t dev_phy_id = id_mgr->GetGpuPhyIdFromThrdId(first_identity_node->thrd_id());
    const auto& fw_nodes = global_dev_phy_id2fw_nodes.at(std::make_pair(machine_id, dev_phy_id));
    const Shape& identity_time_shape =
        *first_identity_node->GetProducedRegst("out")->data_regst_time_shape();
    ForEachDeviceSrcUntrainableNode(fw_nodes, [&](CompTaskNode* node) {
      std::shared_ptr<RegstDesc> regst_desc = node->GetProducedRegst("out");
      if (!regst_desc) { return; }
      const Shape& time_shape = *regst_desc->data_regst_time_shape();
      if (!time_shape.Containing(identity_time_shape)) { return; }
      CHECK_EQ(time_shape.elem_cnt() % identity_time_shape.elem_cnt(), 0);
      int regst_desc_num = time_shape.elem_cnt() / identity_time_shape.elem_cnt();
      RegstDesc* ctrl_regst_desc = node->BuildCtrlRegstDesc(first_identity_node);
      ctrl_regst_desc->UpdtMinRegstNumIfNeed(regst_desc_num);
      ctrl_regst_desc->UpdtMaxRegstNumIfNeed(regst_desc_num);
      ctrl_regst_desc->mut_regst_desc_type()->mutable_ctrl_regst_desc()->set_returned_regst_num(
          regst_desc_num);
    });
  }
}

void TaskGraph::EnableMemSharingInReduceStruct() {
  auto GetSuccReduceTaskNode = [](TaskNode* pred) {
    std::vector<TaskNode*> nodes;
    pred->ForEachNodeOnOutEdge([&](TaskNode* succ) {
      if (dynamic_cast<ReduceCompTaskNodeIf*>(succ) != nullptr) { nodes.push_back(succ); }
    });
    return nodes;
  };

  HashSet<TaskNode*> has_enabled_nodes;

  auto CollectReduceTaskNode = [&](TaskNode* from) {
    std::list<TaskNode*> nodes;
    nodes.push_back(from);
    TaskNode* pred = from;
    while (true) {
      std::vector<TaskNode*> succ_reduce_nodes = GetSuccReduceTaskNode(pred);
      if (succ_reduce_nodes.size() != 1) { break; }
      TaskNode* succ_reduce_node = succ_reduce_nodes.front();
      if (has_enabled_nodes.find(succ_reduce_node) != has_enabled_nodes.end()) { break; }
      nodes.push_back(succ_reduce_node);
      pred = succ_reduce_node;
    }
    return nodes;
  };

  auto CalcModelSize = [](ReduceIdentityCompTaskNode* node) {
    return InferRegstSize(*node->produced_regsts().at("out").get());
  };

  ForEachNode([&](TaskNode* node) {
    ReduceIdentityCompTaskNode* identity_node = dynamic_cast<ReduceIdentityCompTaskNode*>(node);
    if (!identity_node) { return; }
    if (identity_node->parallel_ctx()->policy() != ParallelPolicy::kDataParallel) { return; }
    if (identity_node->device_type() != DeviceType::kGPU) { return; }
    if (identity_node->parallel_ctx()->parallel_num() < 2) { return; }
    std::list<TaskNode*> reduce_task_nodes = CollectReduceTaskNode(identity_node);

    const int64_t mem_shared_id = Global<IDMgr>::Get()->NewMemSharedId();
    const int64_t mem_size = CalcModelSize(identity_node);
    ReduceMemSharingCtx ctx(mem_size, mem_shared_id);
    for (TaskNode* reduce_node : reduce_task_nodes) {
      auto reduce_task_node_if = dynamic_cast<ReduceCompTaskNodeIf*>(reduce_node);
      CHECK_NOTNULL(reduce_task_node_if);
      reduce_task_node_if->EnableMemSharingInReduce(ctx);
      has_enabled_nodes.insert(reduce_node);
    }
  });
}

void TaskGraph::EnableMemSharingAfterAllManualSetForMdUpdt() {
  ForEachNode([&](TaskNode* node) {
    auto* updt = dynamic_cast<NormalMdUpdtCompTaskNode*>(node);
    if (!updt) { return; }
    updt->EnableMemSharingBetweenFirstInAndProcessedMdDiffRegst();
  });
}

void TaskGraph::EnableMemSharingInVariableOp() {
  ForEachNode([&](TaskNode* node) {
    if (node->exec_gph().node_num() != 1) { return; }
    auto* variable_op = dynamic_cast<const VariableOp*>(node->exec_gph().SoleNode()->op().get());
    if (variable_op == nullptr) { return; }
    std::string model_bn = variable_op->op_conf().variable_conf().model_name();
    auto* fw_task_node = dynamic_cast<NormalForwardCompTaskNode*>(node);
    auto* bw_task_node = dynamic_cast<NormalBackwardCompTaskNode*>(node);
    if (fw_task_node != nullptr) {
      const LogicalBlobId& lbi = variable_op->BnInOp2Lbi(model_bn);
      RegstDesc* model_regst = fw_task_node->GetSoleConsumedRegst("model").get();
      CHECK_EQ(model_regst->min_register_num(), 1);
      CHECK_EQ(model_regst->max_register_num(), 1);
      model_regst->set_enable_mem_sharing(true);
      if (model_regst->mem_shared_id() == -1) {
        model_regst->set_mem_shared_id(Global<IDMgr>::Get()->NewMemSharedId());
        model_regst->set_mem_shared_offset(0);
      }
      RegstDesc* out_regst = fw_task_node->GetProducedRegst("out").get();
      CHECK_EQ(out_regst->min_register_num(), 1);
      CHECK_EQ(out_regst->max_register_num(), 1);
      CHECK_EQ(out_regst->NumOfLbi(), 1);
      out_regst->set_enable_mem_sharing(true);
      out_regst->set_mem_shared_id(model_regst->mem_shared_id());
      out_regst->set_mem_shared_offset(model_regst->mem_shared_offset()
                                       + model_regst->ByteOffsetInPackedBlobDescBody(lbi));
      variable_op->set_is_fw_inplace(true);
    } else if (bw_task_node != nullptr) {
      const LogicalBlobId& lbi = variable_op->BnInOp2Lbi(GenDiffBn(model_bn));
      RegstDesc* model_diff_regst = bw_task_node->GetProducedRegst("model_diff").get();
      CHECK_EQ(model_diff_regst->min_register_num(), 1);
      CHECK_EQ(model_diff_regst->max_register_num(), 1);
      model_diff_regst->set_enable_mem_sharing(true);
      if (model_diff_regst->mem_shared_id() == -1) {
        model_diff_regst->set_mem_shared_id(Global<IDMgr>::Get()->NewMemSharedId());
        model_diff_regst->set_mem_shared_offset(0);
      }
      RegstDesc* out_diff_regst = bw_task_node->GetSoleConsumedRegst("out_diff").get();
      if (out_diff_regst->min_register_num() != 1) { return; }
      if (out_diff_regst->max_register_num() != 1) { return; }
      if (out_diff_regst->NumOfLbi() != 1) { return; }
      out_diff_regst->set_enable_mem_sharing(true);
      out_diff_regst->set_mem_shared_id(model_diff_regst->mem_shared_id());
      out_diff_regst->set_mem_shared_offset(
          model_diff_regst->mem_shared_offset()
          + model_diff_regst->ByteOffsetInPackedBlobDescBody(lbi));
      variable_op->set_is_bw_inplace(true);
    } else {
      // do nothing
    }
  });
}

void TaskGraph::EnableInplaceMemSharing() {
  AcyclicTopoForEachNode([&](TaskNode* node) {
    if (node->exec_gph().node_num() != 1) { return; }
    const Operator* op = node->exec_gph().SoleNode()->op().get();
    auto* fw_task_node = dynamic_cast<NormalForwardCompTaskNode*>(node);
    auto* bw_task_node = dynamic_cast<NormalBackwardCompTaskNode*>(node);
    RegstDesc* input_regst = nullptr;
    RegstDesc* output_regst = nullptr;
    if (op->IsForwardInplace() && fw_task_node) {
      input_regst = fw_task_node->GetSoleConsumedRegst("in").get();
      output_regst = fw_task_node->GetProducedRegst("out").get();
    } else if (op->IsBackwardInplace() && bw_task_node) {
      input_regst = bw_task_node->GetSoleConsumedRegst(GenDiffBn("out")).get();
      output_regst = bw_task_node->GetProducedRegst(GenDiffBn("in")).get();
    } else {
      // do nothing
      return;
    }
    if (input_regst->consumers().size() != 1) { return; }
    if (input_regst->NumOfLbi() != 1) { return; }
    if (output_regst->NumOfLbi() != 1) { return; }
    if (input_regst->mem_shared_inplace_block_id() == -1) {
      input_regst->set_mem_shared_inplace_block_id(Global<IDMgr>::Get()->NewMemBlockId());
    }
    output_regst->set_mem_shared_inplace_block_id(input_regst->mem_shared_inplace_block_id());
  });
}

void TaskGraph::RmUselessConsumeRelationshipBetweenFwBw() {
  for (TaskNode* task_node : ordered_task_nodes_) {
    auto bw_node = dynamic_cast<NormalBackwardCompTaskNode*>(task_node);
    if (bw_node == nullptr) { continue; }
    bw_node->RmUselessConsumeRelationshipToFw();
  }
}

void TaskGraph::AddOrderCtrlEdgeBetweenCopyAndMdUpdt() {
  for (TaskNode* task_node : ordered_task_nodes_) {
    auto copy_hd_task_node = dynamic_cast<CopyHdTaskNode*>(task_node);
    if (copy_hd_task_node == nullptr) { continue; }
    if (copy_hd_task_node->copy_type() != CopyHdOpConf::H2D) { continue; }
    if (copy_hd_task_node->area_id() != static_cast<int64_t>(kDataForwardArea)
        && copy_hd_task_node->area_id() != static_cast<int64_t>(kBoundaryArea)) {
      continue;
    }
    std::vector<TaskNode*> candidate_nodes;
    auto ForEachNextNode = [&](TaskNode* node,
                               const std::function<void(TaskNode*)>& TryPushNodeToQueue) {
      auto fw_task_node = dynamic_cast<NormalForwardCompTaskNode*>(node);
      if (fw_task_node != nullptr && fw_task_node->logical_node()->HasOpWithModelBlob()) { return; }
      node->ForEachNodeOnOutEdge([&](TaskNode* node_on_out_edge) {
        if (IsForwardTaskType(node_on_out_edge->GetTaskType())) {
          TryPushNodeToQueue(node_on_out_edge);
        }
      });
    };
    auto HandlerAddCandidate = [&](TaskNode* node) {
      auto fw_task_node = dynamic_cast<NormalForwardCompTaskNode*>(node);
      if (fw_task_node != nullptr && fw_task_node->logical_node()->HasOpWithModelBlob()
          && fw_task_node->parallel_ctx()->parallel_num() > 1
          && fw_task_node->parallel_ctx()->policy() == kDataParallel) {
        candidate_nodes.push_back(node);
      }
    };
    BfsForEachNode({task_node}, ForEachNextNode, HandlerAddCandidate);
    std::sort(candidate_nodes.begin(), candidate_nodes.end(),
              [](const TaskNode* a, const TaskNode* b) {
                return a->order_in_graph() < b->order_in_graph();
              });
    int64_t last_chain_id = -1;
    for (TaskNode* candidate_node : candidate_nodes) {
      if (candidate_node->chain_id() != last_chain_id) {
        last_chain_id = candidate_node->chain_id();
        candidate_node->ForEachNodeOnInEdge([&](TaskNode* node_on_in_edge) {
          if (IsMdUpdtTaskType(node_on_in_edge->GetTaskType())) {
            RegstDesc* ctrl_regst = task_node->BuildCtrlRegstDesc(node_on_in_edge);
            RegstDesc* copy_out_regst = copy_hd_task_node->GetProducedRegst("copy_out").get();
            int64_t piece_num_in_batch = Global<JobDesc>::Get()->NumOfPiecesInBatch();
            ctrl_regst->UpdtMinRegstNumIfNeed(copy_out_regst->min_register_num()
                                              + piece_num_in_batch - 1);
            CtrlRegstDesc* ctrl_regst_desc =
                ctrl_regst->mut_regst_desc_type()->mutable_ctrl_regst_desc();
            ctrl_regst_desc->set_reliant_regst_desc_id(copy_out_regst->regst_desc_id());
            ctrl_regst_desc->set_returned_regst_num(piece_num_in_batch);
          }
        });
      }
    }
  }
}

void TaskGraph::SetAreaIdForNewNodes(const LogicalNode* src_logical,
                                     const LogicalNode* dst_logical) {
  CHECK(src_logical != nullptr && dst_logical != nullptr);
  ForEachNode([&](TaskNode* node) {
    if (node->area_id() != static_cast<int64_t>(kInvalidArea)) return;
    if (src_logical->GetAreaId() == dst_logical->GetAreaId()) {
      node->set_area_id(src_logical->GetAreaId());
    } else {
      node->set_area_id(static_cast<int64_t>(kBoundaryArea));
    }
  });
}

#define DEFINE_BLD_SUB_TASK_GRAPH_METHOD(method_name) \
  void TaskGraph::method_name BLD_SUB_TSK_GPH_MTHD_ARGS()

DEFINE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphByBoxing) {
  std::vector<TaskNode*>* sorted_out_box = nullptr;
  if (logical2sorted_out_box->find(src_logical) == logical2sorted_out_box->end()) {
    BuildOutBoxing(src_logical, sorted_src_comp_tasks, &((*logical2sorted_out_box)[src_logical]),
                   MutBufTask, AllocateCpuThrdIdEvenly);
  }
  sorted_out_box = &(logical2sorted_out_box->at(src_logical));

  std::vector<TaskNode*>* sorted_in_box = nullptr;
  if (logical2sorted_in_box->find(dst_logical) == logical2sorted_in_box->end()) {
    BuildInBoxing(dst_logical, sorted_dst_comp_tasks, &((*logical2sorted_in_box)[dst_logical]),
                  AllocateCpuThrdIdEvenly);
  }
  sorted_in_box = &(logical2sorted_in_box->at(dst_logical));

  for (TaskNode* src_box : *sorted_out_box) {
    for (TaskNode* dst_box : *sorted_in_box) { ConnectWithCopyCommNetIfNeed(src_box, dst_box); }
  }
}

DEFINE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphByOneToOne) {
  CHECK_EQ(sorted_src_comp_tasks.size(), sorted_dst_comp_tasks.size());
  FOR_RANGE(size_t, i, 0, sorted_src_comp_tasks.size()) {
    CompTaskNode* src = sorted_src_comp_tasks[i];
    CompTaskNode* dst = sorted_dst_comp_tasks[i];
    BuildTaskPath(src, dst, MutBufTask, (dst->GetTaskType() != TaskType::kMdSave));
  }
}

DEFINE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphByBroadcastToBroadcast) {
  CHECK(src_logical->SoleOp()->op_conf().has_tick_conf());
  HashMap<size_t, CompTaskNode*> machine_id2last_src_task;
  HashMap<std::pair<int64_t, int64_t>, CompTaskNode*> global_thrd_id2src_task;
  auto GlobalThrdId4TaskNode = [](TaskNode* task_node) -> std::pair<int64_t, int64_t> {
    return std::make_pair(task_node->machine_id(), task_node->thrd_id());
  };
  for (CompTaskNode* src_node : sorted_src_comp_tasks) {
    machine_id2last_src_task[src_node->machine_id()] = src_node;
    global_thrd_id2src_task[GlobalThrdId4TaskNode(src_node)] = src_node;
  }
  HashMap<std::pair<int64_t, int64_t>, CompTaskNode*> global_thrd_id2dst_task;
  for (CompTaskNode* dst_node : sorted_dst_comp_tasks) {
    global_thrd_id2dst_task[GlobalThrdId4TaskNode(dst_node)] = dst_node;
  }
  auto GetSrcNode = [&](const std::pair<int64_t, int64_t>& global_thrd_id) -> CompTaskNode* {
    const auto& src_task_it = global_thrd_id2src_task.find(global_thrd_id);
    if (src_task_it != global_thrd_id2src_task.end()) { return src_task_it->second; }
    const auto& m_src_task_it = machine_id2last_src_task.find(global_thrd_id.first);
    if (m_src_task_it != machine_id2last_src_task.end()) { return m_src_task_it->second; }
    return machine_id2last_src_task.begin()->second;
  };
  for (const auto& pair : global_thrd_id2dst_task) {
    BuildTaskPath(GetSrcNode(pair.first), pair.second, MutBufTask, true);
  }
}

DEFINE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphBySelectOneSourceToSoleSink) {
  CHECK_EQ(sorted_dst_comp_tasks.size(), 1);
  CompTaskNode* sole_dst_comp_task = sorted_dst_comp_tasks.front();
  CompTaskNode* selected_src_comp_task = nullptr;
  bool is_same_machine = false;
  auto UpdateSelected = [&](CompTaskNode* node) {
    selected_src_comp_task = node;
    is_same_machine = (node->machine_id() == sole_dst_comp_task->machine_id());
  };
  for (CompTaskNode* src_comp_task : sorted_src_comp_tasks) {
    if (selected_src_comp_task == nullptr) {
      UpdateSelected(src_comp_task);
      continue;
    }
    if (src_comp_task->machine_id() == sole_dst_comp_task->machine_id()) {
      if (is_same_machine == false) {
        UpdateSelected(src_comp_task);
        continue;
      }
      if (src_comp_task->thrd_id() == sole_dst_comp_task->thrd_id()) {
        UpdateSelected(src_comp_task);
        break;
      }
    }
  }
  CHECK_NOTNULL(selected_src_comp_task);
  BldSubTskGphByOneToOne(nullptr, nullptr, {selected_src_comp_task}, sorted_dst_comp_tasks, nullptr,
                         nullptr, MutBufTask, AllocateCpuThrdIdEvenly);
}

DEFINE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphByReduceScatter2ReduceAdd) {
  const LogicalNode* src_logical_node = sorted_src_comp_tasks.front()->logical_node();
  const auto& pd = src_logical_node->parallel_desc();
  bool has_local_reduce =
      pd->sorted_machine_ids().size() > 1 && pd->device_num_of_each_machine() > 1;
  const LogicalNode* pred_src_logical_node = src_logical_node->SoleInEdge()->src_node();
  bool is_local_reduce =
      has_local_reduce
          ? !(dynamic_cast<const ReduceAddLogicalNode*>(pred_src_logical_node)
              || dynamic_cast<const NcclReduceScatterLogicalNode*>(pred_src_logical_node))
          : false;
  for (CompTaskNode* src_comp_task : sorted_src_comp_tasks) {
    for (CompTaskNode* dst_comp_task : sorted_dst_comp_tasks) {
      if (has_local_reduce) {
        if (is_local_reduce) {
          if (src_comp_task->machine_id() == dst_comp_task->machine_id()) {
            BuildTaskPath(src_comp_task, dst_comp_task, MutBufTask, false);
          }
        } else {
          if (src_comp_task->parallel_id() % pd->device_num_of_each_machine()
              == dst_comp_task->parallel_id() % pd->device_num_of_each_machine()) {
            BuildTaskPath(src_comp_task, dst_comp_task, MutBufTask, false);
          }
        }
      } else {
        BuildTaskPath(src_comp_task, dst_comp_task, MutBufTask, false);
      }
    }
  }
}

DEFINE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphByReduceAdd2ReduceGather) {
  const auto& pd = sorted_src_comp_tasks.front()->logical_node()->parallel_desc();
  bool has_local_reduce =
      pd->sorted_machine_ids().size() > 1 && pd->device_num_of_each_machine() > 1;
  for (CompTaskNode* src_comp_task : sorted_src_comp_tasks) {
    for (CompTaskNode* dst_comp_task : sorted_dst_comp_tasks) {
      if (has_local_reduce) {
        if (src_comp_task->parallel_id() % pd->device_num_of_each_machine()
            == dst_comp_task->parallel_id() % pd->device_num_of_each_machine()) {
          BuildTaskPath(src_comp_task, dst_comp_task, MutBufTask, true);
        }
      } else {
        BuildTaskPath(src_comp_task, dst_comp_task, MutBufTask, true);
      }
    }
  }
}

DEFINE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphByAllToAll) {
  for (CompTaskNode* src_comp_task : sorted_src_comp_tasks) {
    for (CompTaskNode* dst_comp_task : sorted_dst_comp_tasks) {
      Connect<TaskNode>(src_comp_task, NewEdge(), dst_comp_task);
    }
  }
}

DEFINE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphByReduceGather2ReduceGather) {
  const auto& pd = sorted_src_comp_tasks.front()->logical_node()->parallel_desc();
  CHECK_GT(pd->device_num_of_each_machine(), 1);
  CHECK_GT(pd->sorted_machine_ids().size(), 1);
  for (CompTaskNode* src_comp_task : sorted_src_comp_tasks) {
    for (CompTaskNode* dst_comp_task : sorted_dst_comp_tasks) {
      if (src_comp_task->machine_id() == dst_comp_task->machine_id()) {
        BuildTaskPath(src_comp_task, dst_comp_task, MutBufTask, true);
      }
    }
  }
}

void TaskGraph::BuildTaskPath(
    CompTaskNode* src, CompTaskNode* dst,
    std::function<TaskNode**(CompTaskNode* src, int64_t machine_id, int32_t mem_zone_id)>
        MutBufTask,
    bool use_buf_task_node) {
  CHECK_NE(src, dst);
  auto GetBufTask = [&](int64_t machine_id, int32_t mem_zone_id) {
    return *MutBufTask(src, machine_id, mem_zone_id);
  };
  auto SetBufTask = [&](int64_t machine_id, int32_t mem_zone_id, TaskNode* new_val) {
    TaskNode** cur_val = MutBufTask(src, machine_id, mem_zone_id);
    if (*cur_val == nullptr) {
      *cur_val = new_val;
    } else {
      CHECK_EQ(*cur_val, new_val);
    }
    return new_val;
  };

  TaskNode* cur_node = src;
  while (cur_node->machine_id() != dst->machine_id()
         || cur_node->MemZoneId121() != dst->MemZoneId121()) {
    cur_node = BuildTaskStep(cur_node, dst, GetBufTask, SetBufTask, use_buf_task_node);
  }
  Connect<TaskNode>(cur_node, NewEdge(), dst);
}

TaskNode* TaskGraph::BuildTaskStep(
    TaskNode* cur_node, TaskNode* dst,
    std::function<TaskNode*(int64_t machine_id, int32_t mem_zone_id)> GetBufTask,
    std::function<TaskNode*(int64_t machine_id, int32_t mem_zone_id, TaskNode*)> SetBufTask,
    bool use_buf_task_node) {
  int32_t cpu_mem_zone_id = Global<IDMgr>::Get()->CpuMemZoneId();
  int32_t next_mem_zone_id = -1;
  TaskNode* next_node = nullptr;
  if (cur_node->MemZoneId121() != cpu_mem_zone_id) {
    next_mem_zone_id = cpu_mem_zone_id;
    if (!use_buf_task_node || !(next_node = GetBufTask(cur_node->machine_id(), next_mem_zone_id))) {
      next_node = AddCopyD2HTaskFrom(cur_node);
      Connect<TaskNode>(cur_node, NewEdge(), next_node);
    }
  } else if (cur_node->machine_id() == dst->machine_id()) {
    next_mem_zone_id = dst->MemZoneId121();
    if (!use_buf_task_node || !(next_node = GetBufTask(cur_node->machine_id(), next_mem_zone_id))) {
      next_node = AddCopyH2DTaskTo(dst);
      Connect<TaskNode>(cur_node, NewEdge(), next_node);
    }
  } else if (cur_node->machine_id() != dst->machine_id()) {
    next_mem_zone_id = cpu_mem_zone_id;
    if (!use_buf_task_node || !(next_node = GetBufTask(dst->machine_id(), next_mem_zone_id))) {
      next_node = AddCopyCommNetTaskBetween(cur_node, dst);
      Connect<TaskNode>(cur_node, NewEdge(), next_node);
    }
  } else {
    UNIMPLEMENTED();
  }
  if (use_buf_task_node) { SetBufTask(next_node->machine_id(), next_mem_zone_id, next_node); }
  return next_node;
}

TaskNode* TaskGraph::AddCopyH2DTaskTo(TaskNode* task) {
  CHECK_EQ(task->device_type(), DeviceType::kGPU);
  CopyHdTaskNode* copy_task = NewNode<CopyHdTaskNode>();
  copy_task->Init(CopyHdOpConf::H2D, task->machine_id(), task->GpuPhyId());
  return copy_task;
}

TaskNode* TaskGraph::AddCopyD2HTaskFrom(TaskNode* task) {
  CHECK_EQ(task->device_type(), DeviceType::kGPU);
  CopyHdTaskNode* copy_task = NewNode<CopyHdTaskNode>();
  copy_task->Init(CopyHdOpConf::D2H, task->machine_id(), task->GpuPhyId());
  return copy_task;
}

TaskNode* TaskGraph::AddCopyCommNetTaskBetween(TaskNode* src, TaskNode* dst) {
  CHECK_NE(src->machine_id(), dst->machine_id());
  CopyCommNetTaskNode* copy_comm_net_task = NewNode<CopyCommNetTaskNode>();
  copy_comm_net_task->Init(dst->machine_id(), src->machine_id());
  return copy_comm_net_task;
}

void TaskGraph::BuildOutBoxing(
    const LogicalNode* logical, const std::vector<CompTaskNode*>& sorted_comp_tasks,
    std::vector<TaskNode*>* sorted_out_box,
    std::function<TaskNode**(CompTaskNode* src, int64_t machine_id, int32_t mem_zone_id)>
        MutBufTask,
    std::function<int64_t(const TaskNode*)> AllocateCpuThrdIdEvenly) {
  std::map<int64_t, std::vector<TaskNode*>> machine_id2bound_task;
  for (CompTaskNode* comp_task : sorted_comp_tasks) {
    TaskNode* task = comp_task;
    if (task->device_type() == DeviceType::kGPU) {
      TaskNode** buf_task =
          MutBufTask(comp_task, comp_task->machine_id(), Global<IDMgr>::Get()->CpuMemZoneId());
      if ((*buf_task) == nullptr) {
        task = AddCopyD2HTaskFrom(comp_task);
        Connect<TaskNode>(comp_task, NewEdge(), task);
        *buf_task = task;
      } else {
        task = *buf_task;
      }
    }
    machine_id2bound_task[task->machine_id()].push_back(task);
  }
  for (const auto& pair : machine_id2bound_task) {
    OutBoxingTaskNode* boxing_task = NewNode<OutBoxingTaskNode>();
    boxing_task->set_machine_id(pair.second.front()->machine_id());
    boxing_task->set_thrd_id(AllocateCpuThrdIdEvenly(boxing_task));
    for (TaskNode* task : pair.second) { Connect<TaskNode>(task, NewEdge(), boxing_task); }
    sorted_out_box->push_back(boxing_task);
  }
}

void TaskGraph::BuildInBoxing(const LogicalNode* logical,
                              const std::vector<CompTaskNode*>& sorted_comp_tasks,
                              std::vector<TaskNode*>* sorted_in_box,
                              std::function<int64_t(const TaskNode*)> AllocateCpuThrdIdEvenly) {
  std::map<int64_t, std::vector<TaskNode*>> machine_id2bound_task;
  for (CompTaskNode* comp_task : sorted_comp_tasks) {
    TaskNode* task = comp_task;
    if (task->device_type() == DeviceType::kGPU) {
      task = AddCopyH2DTaskTo(comp_task);
      Connect<TaskNode>(task, NewEdge(), comp_task);
    }
    machine_id2bound_task[task->machine_id()].push_back(task);
  }
  for (const auto& pair : machine_id2bound_task) {
    InBoxingTaskNode* boxing_task = NewNode<InBoxingTaskNode>();
    boxing_task->set_machine_id(pair.second.front()->machine_id());
    boxing_task->set_thrd_id(AllocateCpuThrdIdEvenly(boxing_task));
    for (TaskNode* task : pair.second) { Connect<TaskNode>(boxing_task, NewEdge(), task); }
    sorted_in_box->push_back(boxing_task);
  }
}

void TaskGraph::ConnectWithCopyCommNetIfNeed(TaskNode* src, TaskNode* dst) {
  if (src->machine_id() == dst->machine_id()) {
    Connect(src, NewEdge(), dst);
  } else {
    TaskNode* copy_comm_net_task = AddCopyCommNetTaskBetween(src, dst);
    Connect<TaskNode>(src, NewEdge(), copy_comm_net_task);
    Connect<TaskNode>(copy_comm_net_task, NewEdge(), dst);
  }
}

bool IsBackEdge(TaskNode* src, TaskNode* dst) {
  return src->GetTaskType() == TaskType::kNormalMdUpdt
         && (dst->GetTaskType() == TaskType::kNormalForward
             || dst->GetTaskType() == TaskType::kNormalBackward);
}

}  // namespace oneflow
