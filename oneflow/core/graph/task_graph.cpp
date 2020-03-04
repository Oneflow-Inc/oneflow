#include "oneflow/core/graph/task_graph.h"
#include "oneflow/core/graph/normal_forward_compute_task_node.h"
#include "oneflow/core/graph/chain_graph.h"
#include "oneflow/core/graph/boxing_task_node.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/graph/reduce_add_compute_task_node.h"
#include "oneflow/core/graph/inplace_lbi_graph.h"
#include "oneflow/core/register/runtime_blob_desc.h"
#include "oneflow/core/job/thrd_id_generator.h"
#include "oneflow/core/graph/reduce_identity_task_node.h"
#include "oneflow/core/operator/variable_op.h"
#include "oneflow/core/operator/constant_op.h"
#include "oneflow/core/operator/user_op_util.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/graph/boxing/sub_task_graph_builder_context.h"
#include "oneflow/core/graph/boxing/sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/chain_sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/nccl_boxing_sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/slice_boxing_sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/sub_task_graph_builder_util.h"

namespace oneflow {

namespace {

bool IsInterfaceTask(const TaskNode* node) {
  const auto* comp_task_node = dynamic_cast<const CompTaskNode*>(node);
  if (comp_task_node == nullptr) { return false; }
  if (comp_task_node->logical_node()->op_vec().size() != 1) { return false; }
  auto op_type_case = comp_task_node->logical_node()->SoleOp()->op_conf().op_type_case();
  return IsClassRegistered<IsInterfaceOpConf4OpTypeCase>(op_type_case);
}

bool IsConnectToTickOp(const TaskNode* node) {
  const auto* comp_task_node = dynamic_cast<const CompTaskNode*>(node);
  if (comp_task_node == nullptr) { return false; }
  if (comp_task_node->logical_node()->op_vec().size() != 1) { return false; }
  const Operator* op = comp_task_node->logical_node()->SoleOp().get();
  if (dynamic_cast<const VariableOp*>(op) != nullptr) { return true; }
  if (dynamic_cast<const ConstantOp*>(op) != nullptr) { return true; }
  return false;
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
    // TODO: update method for fw bw split
    // const auto* fw_logical_node = dynamic_cast<const ForwardLogicalNode*>(node->logical_node());
    // return fw_logical_node->bw_node() != nullptr;
    return false;
  };
  for (NormalForwardCompTaskNode* fw_node : fw_nodes) {
    if (IsSourceTaskNode(fw_node) && !HasBwNode(fw_node)) { Handler(fw_node); }
  }
}

std::function<TaskNode*(const std::string&)> MakeGetterTaskNode4SoleOpName(
    const HashSet<TaskNode*>& task_nodes) {
  auto op_name2task_nodes = std::make_shared<HashMap<std::string, HashSet<TaskNode*>>>();
  for (TaskNode* task_node : task_nodes) {
    if (task_node->exec_gph().node_num() == 1) {
      ExecNode* exec_node = task_node->exec_gph().SoleNode();
      CHECK((*op_name2task_nodes)[exec_node->op()->op_name()].emplace(task_node).second);
    }
  }
  return [op_name2task_nodes](const std::string& op_name) -> TaskNode* {
    const auto& iter = op_name2task_nodes->find(op_name);
    if (iter == op_name2task_nodes->end()) { return nullptr; }
    if (iter->second.size() > 1) { return nullptr; }
    return *iter->second.begin();
  };
}

bool IsLbiOnTaskEdge(const TaskEdge* edge, const LogicalBlobId& lbi) {
  for (const auto& regst_desc : edge->GetRegsts()) {
    if (regst_desc->HasLbi(lbi)) { return true; }
  }
  return false;
}

std::function<bool(const LogicalBlobId&, const std::string&)>
MakePredicatorIsLbiAllConsumersReachable(
    std::function<const TaskNode*(const std::string&)> TaskNode4SoleOpName,
    std::function<bool(const std::string&, const std::string&)> IsOpNameDataOrCtrlReachable) {
  auto IsDataOrCtrlReachable = [IsOpNameDataOrCtrlReachable](const TaskNode* src_node,
                                                             const TaskNode* dst_node) -> bool {
    if (src_node->chain_id() == dst_node->chain_id()
        && src_node->order_in_graph() <= dst_node->order_in_graph()) {
      return true;
    }
    const CompTaskNode* comp_src_node = dynamic_cast<const CompTaskNode*>(src_node);
    if (comp_src_node == nullptr) { return false; }
    if (comp_src_node->logical_node()->op_vec().size() != 1) { return false; }
    const CompTaskNode* comp_dst_node = dynamic_cast<const CompTaskNode*>(dst_node);
    if (comp_dst_node == nullptr) { return false; }
    if (comp_dst_node->logical_node()->op_vec().size() != 1) { return false; }
    return IsOpNameDataOrCtrlReachable(comp_src_node->logical_node()->SoleOp()->op_name(),
                                       comp_dst_node->logical_node()->SoleOp()->op_name());
  };
  return [TaskNode4SoleOpName, IsDataOrCtrlReachable](const LogicalBlobId& lbi,
                                                      const std::string& op_name) -> bool {
    const TaskNode* src_task_node = TaskNode4SoleOpName(lbi.op_name());
    const TaskNode* dst_task_node = TaskNode4SoleOpName(op_name);
    size_t out_edges_size = 0;
    size_t reachable_out_edges_size = 0;
    for (TaskEdge* out_edge : src_task_node->out_edges()) {
      if (IsLbiOnTaskEdge(out_edge, lbi)) {
        out_edges_size += 1;
        reachable_out_edges_size += IsDataOrCtrlReachable(out_edge->dst_node(), dst_task_node);
      }
    }
    return out_edges_size > 0 && out_edges_size == reachable_out_edges_size;
  };
}

bool IsInplaceAllowed(
    TaskNode* task_node, const std::vector<std::string>& bns,
    const std::function<const TaskNode*(const std::string&)>& TaskNode4SoleOpName) {
  if (task_node->exec_gph().node_num() != 1) { return false; }
  const auto& exec_node = *task_node->exec_gph().SoleNode();
  for (const auto& bn : bns) {
    // TaskNode for bn is not nullptr if it's on the same device with `task_node`
    if (TaskNode4SoleOpName(exec_node.op()->BnInOp2Lbi(bn).op_name()) == nullptr) { return false; }
    const RegstDesc& regst_desc = *exec_node.RegstDesc4BnInOp(bn);
    if (regst_desc.NumOfLbi() != 1) { return false; }
  }
  const BlobDesc* first_blob = nullptr;
  for (const auto& bn : bns) {
    const BlobDesc* blob_desc = exec_node.RegstDesc4BnInOp(bn)->SoleBlobDesc();
    if (first_blob == nullptr) {
      first_blob = blob_desc;
    } else {
      if (!(first_blob->shape() == blob_desc->shape()
            && first_blob->data_type() == blob_desc->data_type())) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace

TaskGraph::TaskGraph(std::unique_ptr<const LogicalGraph>&& logical_gph) {
  logical_gph_ = std::move(logical_gph);
  HashMap<const LogicalNode*, std::vector<CompTaskNode*>> logical2sorted_comp_tasks;
  HashMap<const LogicalNode*, std::vector<TaskNode*>> logical2sorted_in_box;
  HashMap<const LogicalNode*, std::vector<TaskNode*>> logical2sorted_out_box;
  HashMap<CompTaskNode*, HashMap<int64_t, std::vector<TaskNode*>>> buf_task;
  auto MutBufTask = [&](CompTaskNode* task_node, int64_t machine_id, int32_t mem_zone_id) {
    auto& buf_vec = buf_task[task_node][machine_id];
    if (buf_vec.empty()) { buf_vec.assign(Global<ResourceDesc>::Get()->MemZoneNum(), nullptr); }
    return &(buf_vec.at(mem_zone_id));
  };

  std::vector<int64_t> cpu_device_offset(Global<ResourceDesc>::Get()->TotalMachineNum(), 0);
  auto AllocateCpuThrdIdEvenly = [&](const TaskNode* task_node) {
    CHECK(!task_node->IsIndependent());
    int64_t ret = -1;
    int64_t& offset = cpu_device_offset.at(task_node->machine_id());
    ret = Global<IDMgr>::Get()->GetCpuDeviceThrdId(offset);
    offset = (offset + 1) % Global<ResourceDesc>::Get()->CpuDeviceNum();
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

  GenerateIndependentThrdId(machine_persistence_task_vec);
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
  if (Global<ResourceDesc>::Get()->enable_debug_mode()) { ToDotWithAutoFilePath(); }
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

void TaskGraph::GenerateIndependentThrdId(
    const std::vector<std::pair<int64_t, CompTaskNode*>>& persistence_nodes) {
  std::vector<std::pair<int64_t, TaskType>> machine_task_type_vec;
  for (auto pair : persistence_nodes) {
    machine_task_type_vec.emplace_back(std::make_pair(pair.first, pair.second->GetTaskType()));
  }

  ThrdIdGenerator generator(machine_task_type_vec, Global<IDMgr>::Get()->BaseIndependentThrdId());
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

void TaskGraph::EnableInplaceMemSharingInReduceStruct() {
  auto GetSuccReduceTaskNode = [](TaskNode* pred) {
    std::vector<TaskNode*> nodes;
    pred->ForEachNodeOnOutDataEdge([&](TaskNode* succ) {
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
    if (identity_node->device_type() != DeviceType::kGPU) { return; }
    if (identity_node->parallel_ctx()->parallel_num() < 2) { return; }
    std::list<TaskNode*> reduce_task_nodes = CollectReduceTaskNode(identity_node);

    const int64_t mem_block_id = Global<IDMgr>::Get()->NewMemBlockId();
    const int64_t mem_size = CalcModelSize(identity_node);
    ReduceMemSharingCtx ctx(mem_size, mem_block_id);
    for (TaskNode* reduce_node : reduce_task_nodes) {
      auto reduce_task_node_if = dynamic_cast<ReduceCompTaskNodeIf*>(reduce_node);
      CHECK_NOTNULL(reduce_task_node_if);
      reduce_task_node_if->EnableMemSharingInReduce(ctx);
      has_enabled_nodes.insert(reduce_node);
    }
  });
}

void TaskGraph::GetInplaceOpBlobArgList(
    InplaceObasInfo* obas_info, const HashSet<TaskNode*>& dev_nodes,
    const std::function<const TaskNode*(const std::string&)>& TaskNode4OpName) const {
  auto AddMutableInplaceArgPair = [&](TaskNode* node, const std::string& ibn,
                                      const std::string& obn, const std::string& op_name) {
    if (IsInplaceAllowed(node, {ibn, obn}, TaskNode4OpName)) {
      auto* pair = obas_info->mut_inplace_oba_pairs.mutable_pair()->Add();
      *pair->mutable_first() = GenOpBlobArg(op_name, ibn);
      *pair->mutable_second() = GenOpBlobArg(op_name, obn);
    }
  };
  auto AddConstInplaceArgPair = [&](TaskNode* node, const std::string& ibn, const std::string& obn,
                                    const std::string& op_name) {
    if (IsInplaceAllowed(node, {ibn, obn}, TaskNode4OpName)) {
      auto* pair = obas_info->con_inplace_oba_pairs.mutable_pair()->Add();
      *pair->mutable_first() = GenOpBlobArg(op_name, ibn);
      *pair->mutable_second() = GenOpBlobArg(op_name, obn);
    }
  };

  for (TaskNode* task_node : dev_nodes) {
    if (task_node->exec_gph().node_num() != 1) { continue; }
    const auto& op = *task_node->exec_gph().SoleNode()->op();
    for (const std::string& ibn : op.input_bns()) {
      if (op.InputBlobModifier4Ibn(ibn).is_mutable()) {
        CHECK(IsInplaceAllowed(task_node, {ibn}, TaskNode4OpName));
        *obas_info->mut_in_obas.mutable_oba()->Add() = GenOpBlobArg(op.op_name(), ibn);
      }
    }
    for (const std::string& obn : op.output_bns()) {
      const auto& obn_modifier = op.OutputBlobModifier4Obn(obn);
      if (obn_modifier.has_mutable_inplace_ibn()) {
        AddMutableInplaceArgPair(task_node, obn_modifier.mutable_inplace_ibn(), obn, op.op_name());
      } else if (obn_modifier.has_const_inplace_ibn()) {
        AddConstInplaceArgPair(task_node, obn_modifier.const_inplace_ibn(), obn, op.op_name());
      }
    }

    if (op.op_conf().has_user_conf()) {
      const OpContext* op_ctx = task_node->exec_gph().SoleNode()->op_context();
      const UserOpCtx* user_op_ctx = static_cast<const UserOpCtx*>(op_ctx);
      for (const auto& pair : user_op_ctx->mut_inplace_obn2ibn) {
        AddMutableInplaceArgPair(task_node, pair.second, pair.first, op.op_name());
      }
      for (const auto& pair : user_op_ctx->con_inplace_obn2ibn) {
        AddConstInplaceArgPair(task_node, pair.second, pair.first, op.op_name());
      }
    }
  }
}

void TaskGraph::GetSafeInplaceOpBlobArgList(
    InplaceObasInfo* safe_obas_info, const HashSet<TaskNode*>& dev_nodes,
    std::function<bool(const std::string&, const std::string&)> IsOpNameDataOrCtrlReachable) const {
  auto TaskNode4SoleOpName = MakeGetterTaskNode4SoleOpName(dev_nodes);
  InplaceObasInfo obas_info;
  GetInplaceOpBlobArgList(&obas_info, dev_nodes, TaskNode4SoleOpName);
  auto Op4OpName = [&](const std::string& op_name) -> const Operator* {
    return TaskNode4SoleOpName(op_name)->exec_gph().SoleNode()->op().get();
  };
  auto IsLbiAllConsumersReachable =
      MakePredicatorIsLbiAllConsumersReachable(TaskNode4SoleOpName, IsOpNameDataOrCtrlReachable);
  InplaceLbiGraph origin_graph(obas_info, Op4OpName);
  InplaceLbiGraph safe_graph(*safe_obas_info, Op4OpName);
  origin_graph.ComputeSafeInplaceObns(safe_obas_info, IsLbiAllConsumersReachable);
  if (Global<ResourceDesc>::Get()->enable_debug_mode()) {
    origin_graph.ToDotWithFilePath(
        JoinPath("dot", "InplaceLbiGraph", GlobalJobDesc().job_name() + "_origin.dot"));
    safe_graph.ToDotWithFilePath(
        JoinPath("dot", "InplaceLbiGraph", GlobalJobDesc().job_name() + "_safe.dot"));
  }
}

void TaskGraph::SetTaskRegstInplaceInfo(const InplaceObasInfo& obas_info,
                                        const HashSet<TaskNode*>& dev_nodes) const {
  auto TaskNode4SoleOpName = MakeGetterTaskNode4SoleOpName(dev_nodes);
  auto Op4OpName = [&](const std::string& op_name) -> const Operator* {
    return TaskNode4SoleOpName(op_name)->exec_gph().SoleNode()->op().get();
  };
  InplaceLbiGraph inplace_gph(obas_info, Op4OpName);
  inplace_gph.ForEachConnectedComponent([&](const HashSet<const InplaceLbiNode*> inplace_nodes) {
    for (const auto* inplace_node : inplace_nodes) {
      if (inplace_node->in_edges().empty()) { continue; }
      const auto* inplace_edge = inplace_node->SoleInEdge();
      auto* exec_node = TaskNode4SoleOpName(inplace_edge->op().op_name())->exec_gph().SoleNode();
      RegstDesc* in_regst = exec_node->RegstDesc4BnInOp(inplace_edge->ibn());
      RegstDesc* out_regst = exec_node->RegstDesc4BnInOp(inplace_edge->obn());
      out_regst->set_hint_inplace_consumed_regst_desc_id(in_regst->regst_desc_id());
    }
  });
}

void TaskGraph::ForEachGpuDeviceNodes(
    const std::function<void(const HashSet<TaskNode*>& dev_nodes)>& Handler) const {
  HashMap<std::pair<int64_t, int64_t>, HashSet<TaskNode*>> global_dev_phy_id2nodes;
  ForEachNode([&](TaskNode* task_node) {
    if (task_node->device_type() != DeviceType::kGPU) { return; }
    int64_t dev_phy_id = Global<IDMgr>::Get()->GetGpuPhyIdFromThrdId(task_node->thrd_id());
    global_dev_phy_id2nodes[{task_node->machine_id(), dev_phy_id}].emplace(task_node);
  });
  for (const auto& pair : global_dev_phy_id2nodes) { Handler(pair.second); }
}

void TaskGraph::EnableInplaceMemSharing(
    const std::function<bool(const std::string&, const std::string&)>&
        IsOpNameDataOrCtrlReachable) {
  ForEachGpuDeviceNodes([&](const HashSet<TaskNode*>& dev_nodes) {
    InplaceObasInfo safe_inplace_obas_info;
    GetSafeInplaceOpBlobArgList(&safe_inplace_obas_info, dev_nodes, IsOpNameDataOrCtrlReachable);
    SetTaskRegstInplaceInfo(safe_inplace_obas_info, dev_nodes);
  });
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
      node->ForEachNodeOnOutEdge([&](TaskNode* node_on_out_edge) {
        if (IsForwardTaskType(node_on_out_edge->GetTaskType())) {
          TryPushNodeToQueue(node_on_out_edge);
        }
      });
    };
    auto HandlerAddCandidate = [&](TaskNode* node) {
      TODO();  // refactor the following code
      /*
      auto fw_task_node = dynamic_cast<NormalForwardCompTaskNode*>(node);
      if (fw_task_node != nullptr && fw_task_node->logical_node()->HasOpWithModelBlob()
          && fw_task_node->parallel_ctx()->parallel_num() > 1
          && fw_task_node->parallel_ctx()->policy() == kDataParallel) {
        candidate_nodes.push_back(node);
      }
      */
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
            int64_t piece_num_in_batch = GlobalJobDesc().NumOfPiecesInBatch();
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
  if (GlobalJobDesc().use_boxing_v2()) {
    BldSubTskGphByBoxingV2(src_logical, dst_logical, sorted_src_comp_tasks, sorted_dst_comp_tasks,
                           logical2sorted_in_box, logical2sorted_out_box, std::move(MutBufTask),
                           std::move(AllocateCpuThrdIdEvenly));
  } else {
    BldSubTskGphByBoxingV1(src_logical, dst_logical, sorted_src_comp_tasks, sorted_dst_comp_tasks,
                           logical2sorted_in_box, logical2sorted_out_box, std::move(MutBufTask),
                           std::move(AllocateCpuThrdIdEvenly));
  }
}

DEFINE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphByBoxingV1) {
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

DEFINE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphByBoxingV2) {
  const std::vector<LogicalBlobId> lbis = src_logical->GetLbisTo(dst_logical);
  const auto Fallback = [&]() {
    BldSubTskGphByBoxingV1(src_logical, dst_logical, sorted_src_comp_tasks, sorted_dst_comp_tasks,
                           logical2sorted_in_box, logical2sorted_out_box, std::move(MutBufTask),
                           std::move(AllocateCpuThrdIdEvenly));
  };
  if (lbis.size() > 1) {
    Fallback();
  } else {
    CHECK_EQ(lbis.size(), 1);
    const LogicalBlobId& lbi = lbis.front();
    const SbpParallel& src_sbp_parallel =
        Global<OpGraph>::Get()->GetSbpParallel(src_logical->SoleOp()->op_name(), lbi);
    const SbpParallel& dst_sbp_parallel =
        Global<OpGraph>::Get()->GetSbpParallel(dst_logical->SoleOp()->op_name(), lbi);
    const std::shared_ptr<const ParallelDesc>& src_parallel_desc = src_logical->parallel_desc();
    const std::shared_ptr<const ParallelDesc>& dst_parallel_desc = dst_logical->parallel_desc();
    const BlobDesc& blob_desc = Global<OpGraph>::Get()->GetLogicalBlobDesc(lbi);
    SubTskGphBuilderCtx ctx(this);
    std::vector<std::shared_ptr<SubTskGphBuilder>> builders;
    builders.emplace_back(new NcclBoxingSubTskGphBuilder());
    builders.emplace_back(new SliceBoxingSubTskGphBuilder());
    Maybe<void> status = TRY(ChainSubTskGphBuilder(builders).Build(
        &ctx, sorted_src_comp_tasks, sorted_dst_comp_tasks, *src_parallel_desc, *dst_parallel_desc,
        lbi, blob_desc, src_sbp_parallel, dst_sbp_parallel));
    if (!status.IsOk()) {
      if (SubTskGphBuilderUtil::IsErrorBoxingNotSupported(*status.error())) {
        Fallback();
      } else {
        UNIMPLEMENTED();
      }
    }
  }
}

DEFINE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphByOneToOne) {
  CHECK_EQ(sorted_src_comp_tasks.size(), sorted_dst_comp_tasks.size());
  FOR_RANGE(size_t, i, 0, sorted_src_comp_tasks.size()) {
    CompTaskNode* src = sorted_src_comp_tasks.at(i);
    CompTaskNode* dst = sorted_dst_comp_tasks.at(i);
    BuildTaskPath(src, dst, MutBufTask, true);
  }
}

DEFINE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphByBroadcastToBroadcast) {
  CHECK_EQ(sorted_dst_comp_tasks.size() % sorted_src_comp_tasks.size(), 0);
  if (sorted_src_comp_tasks.size() == sorted_dst_comp_tasks.size()) {
    FOR_RANGE(size_t, i, 0, sorted_src_comp_tasks.size()) {
      CompTaskNode* src = sorted_src_comp_tasks.at(i);
      CompTaskNode* dst = sorted_dst_comp_tasks.at(i);
      BuildTaskPath(src, dst, MutBufTask, true);
    }
    return;
  }
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

DEFINE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphByConnectNodeOnSameGpuDevice) {
  for (CompTaskNode* src : sorted_src_comp_tasks) {
    for (CompTaskNode* dst : sorted_dst_comp_tasks) {
      if (src->machine_id() == dst->machine_id() && src->GpuPhyId() == dst->GpuPhyId()) {
        Connect<TaskNode>(src, NewEdge(), dst);
      }
    }
  }
}

DEFINE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphByPartialInLbiConnect) {
  HashSet<LogicalBlobId> lbis;
  for (const auto& obn : src_logical->SoleOp()->output_bns()) {
    lbis.insert(src_logical->SoleOp()->BnInOp2Lbi(obn));
  }
  CHECK_EQ(sorted_src_comp_tasks.size(), 1);
  CHECK_EQ(dst_logical->SoleOp()->input_bns().size(), sorted_dst_comp_tasks.size());
  FOR_RANGE(int, i, 0, sorted_dst_comp_tasks.size()) {
    const auto& lbi = dst_logical->SoleOp()->BnInOp2Lbi(dst_logical->SoleOp()->input_bns().Get(i));
    if (lbis.find(lbi) != lbis.end()) {
      BuildTaskPath(sorted_src_comp_tasks.at(0), sorted_dst_comp_tasks.at(i), MutBufTask, true);
    }
  }
}

DEFINE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphByPartialOutLbiConnect) {
  HashSet<LogicalBlobId> lbis;
  for (const auto& ibn : dst_logical->SoleOp()->input_bns()) {
    lbis.insert(dst_logical->SoleOp()->BnInOp2Lbi(ibn));
  }
  CHECK_EQ(sorted_dst_comp_tasks.size(), 1);
  CHECK_EQ(src_logical->SoleOp()->output_bns().size(), sorted_src_comp_tasks.size());
  FOR_RANGE(int, i, 0, sorted_src_comp_tasks.size()) {
    const auto& lbi = src_logical->SoleOp()->BnInOp2Lbi(src_logical->SoleOp()->output_bns().Get(i));
    if (lbis.find(lbi) != lbis.end()) {
      BuildTaskPath(sorted_src_comp_tasks.at(i), sorted_dst_comp_tasks.at(0), MutBufTask, true);
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
  if (cur_node != dst) { Connect<TaskNode>(cur_node, NewEdge(), dst); }
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
      next_node = TryAddCopyH2DTaskTo(dst);
      if (next_node == nullptr) { next_node = dst; }
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
  if (use_buf_task_node && (next_node != dst)) {
    SetBufTask(next_node->machine_id(), next_mem_zone_id, next_node);
  }
  return next_node;
}

TaskNode* TaskGraph::TryAddCopyH2DTaskTo(TaskNode* task) {
  if (IsInterfaceTask(task)) { return nullptr; }
  if (IsClassRegistered<TickTockTaskType>(task->GetTaskType())) { return nullptr; }
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
      task = TryAddCopyH2DTaskTo(comp_task);
      if (task == nullptr) { task = comp_task; }
      if (task != comp_task) { Connect<TaskNode>(task, NewEdge(), comp_task); }
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

bool IsBackEdge(TaskNode* src, TaskNode* dst) { return false; }

}  // namespace oneflow
