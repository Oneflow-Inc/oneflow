/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/graph/task_graph.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/graph/inplace_lbi_graph.h"
#include "oneflow/core/graph/id_serialization.h"
#include "oneflow/core/register/runtime_blob_desc.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/operator/variable_op.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/graph/normal_forward_compute_task_node.h"
#include "oneflow/core/graph/boxing_identity_task_node.h"
#include "oneflow/core/job/scope.h"
#include "oneflow/core/vm/symbol_storage.h"
#include "oneflow/core/job_rewriter/calculation_pass.h"
#include "oneflow/core/graph/boxing/sub_task_graph_builder_util.h"
#include "oneflow/core/graph/boxing/hierarchical_sub_task_graph_builder_impl.h"
#include "oneflow/core/graph/stream_index_getter_registry_manager.h"

namespace oneflow {

namespace {

bool IsInterfaceTask(const TaskNode* node) {
  const auto* comp_task_node = dynamic_cast<const CompTaskNode*>(node);
  if (comp_task_node == nullptr) { return false; }
  auto op_type_case = comp_task_node->op()->op_conf().op_type_case();
  return IsClassRegistered<int32_t, IsInterfaceOpConf4OpTypeCase>(op_type_case);
}

bool IsConnectToTickOp(const TaskNode* node) {
  const auto* comp_task_node = dynamic_cast<const CompTaskNode*>(node);
  if (comp_task_node == nullptr) { return false; }
  const Operator* op = comp_task_node->op().get();
  if (dynamic_cast<const VariableOp*>(op) != nullptr) { return true; }
  return false;
}

bool IsOptimizerPassOp(const Operator* op) {
  // NOTE(chengcheng): use scope::calculation_pass_name instead of area_id to not merge optimizer
  // ops with fw/bw ops
  if (!op->op_conf().has_scope_symbol_id()) {
    // NOTE(chengcheng): Some system op insert to OpGraph may not set scope_symbol_id, it MUST NOT
    // optimizer subgraph ops.
    return false;
  }
  int64_t scope_symbol_id = op->op_conf().scope_symbol_id();
  CHECK(Global<symbol::Storage<Scope>>::Get()->Has(scope_symbol_id))
      << " Error! op : \n " << op->op_conf().DebugString()
      << " has error scope_symbol_id = " << scope_symbol_id
      << " which cannot find in Global<symbol::Storage<Scope>>::Get()\n";
  const Scope& scope = Global<symbol::Storage<Scope>>::Get()->Get(scope_symbol_id);
  return scope.scope_proto().calculation_pass_name() == kOptimizerPass;
}

bool IsSubsetTickOpConf(const OperatorConf& op_conf) {
  return op_conf.has_src_subset_tick_conf() || op_conf.has_dst_subset_tick_conf();
}

bool IsTickOpConf(const OperatorConf& conf) {
  return IsClassRegistered<int32_t, IsTickTockOpTypeCase>(conf.op_type_case());
}

bool IsSpecialOpNotConsiderMergeInChain(const Operator* op) {
  const OperatorConf& op_conf = op->op_conf();
  if (op_conf.has_variable_conf() || op_conf.has_tick_conf() || op_conf.has_device_tick_conf()
      || op_conf.has_src_subset_tick_conf() || op_conf.has_dst_subset_tick_conf()
      || op_conf.has_source_tick_conf() || op_conf.has_sink_tick_conf()
      || op_conf.has_acc_tick_conf()) {
    return true;
  }
  // NOTE(chengcheng): ONLY nccl_use_compute_stream = false will exclude optimizer pass ops
  if (!Global<ResourceDesc, ForSession>::Get()->nccl_use_compute_stream()
      && IsOptimizerPassOp(op)) {
    return true;
  }
  return false;
}

bool IsTaskNodeProducedResgtHasMultiRegstNum(const TaskNode* node) {
  for (const auto& pair : node->produced_regsts()) {
    if (pair.second->min_register_num() > 1) { return true; }
  }
  return false;
}

bool CanBeMergedInChain(const TaskNode* node) {
  // ONLY the node which is NormalForward and in GPU and NOT variable can be merged.
  if (IsTaskNodeProducedResgtHasMultiRegstNum(node)) { return false; }
  const auto* fw_comp_node = dynamic_cast<const NormalForwardCompTaskNode*>(node);
  if (fw_comp_node == nullptr) { return false; }
  if (fw_comp_node->device_type() != DeviceType::kGPU) { return false; }
  const Operator* op = fw_comp_node->op().get();
  if (IsSpecialOpNotConsiderMergeInChain(op)) { return false; }
  return true;
}

void TraverseConnectedSubGraphMergeInThisChain(TaskNode* this_node, const int64_t this_chain_id) {
  CHECK_NE(this_chain_id, -1);
  CHECK_EQ(this_node->chain_id(), -1);
  // bfs search all node can be merged in this chain
  HashSet<TaskNode*> visited_nodes;
  std::queue<TaskNode*> queued_nodes;
  queued_nodes.push(this_node);
  visited_nodes.insert(this_node);
  while (!queued_nodes.empty()) {
    TaskNode* cur_node = queued_nodes.front();
    queued_nodes.pop();

    CHECK_EQ(cur_node->chain_id(), -1);
    cur_node->set_chain_id(this_chain_id);

    cur_node->ForEachNodeOnInOutEdge([&](TaskNode* next_node) {
      if (visited_nodes.find(next_node) == visited_nodes.end() && CanBeMergedInChain(next_node)
          && this_node->GlobalWorkStreamId() == next_node->GlobalWorkStreamId()) {
        if (next_node->chain_id() == -1) {
          queued_nodes.push(next_node);
          visited_nodes.insert(next_node);
        } else {
          CHECK_EQ(next_node->chain_id(), this_chain_id);
        }
      }
    });
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
    const std::function<const TaskNode*(const std::string&)>& TaskNode4SoleOpName,
    const std::function<bool(const std::string&, const std::string&)>&
        IsOpNameDataOrCtrlReachable) {
  auto IsDataOrCtrlReachable = [IsOpNameDataOrCtrlReachable](const TaskNode* src_node,
                                                             const TaskNode* dst_node) -> bool {
    if (src_node->chain_id() == dst_node->chain_id()
        && src_node->order_in_graph() <= dst_node->order_in_graph()) {
      return true;
    }
    const CompTaskNode* comp_src_node = dynamic_cast<const CompTaskNode*>(src_node);
    if (comp_src_node == nullptr) { return false; }
    const CompTaskNode* comp_dst_node = dynamic_cast<const CompTaskNode*>(dst_node);
    if (comp_dst_node == nullptr) { return false; }
    return IsOpNameDataOrCtrlReachable(comp_src_node->op()->op_name(),
                                       comp_dst_node->op()->op_name());
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
      if (!(first_blob->shape().elem_cnt() == blob_desc->shape().elem_cnt()
            && first_blob->data_type() == blob_desc->data_type())) {
        return false;
      }
    }
  }
  return true;
}

std::unique_ptr<BoxingLogger> CreateBoxingLogger() {
  if (Global<ResourceDesc, ForSession>::Get()->enable_debug_mode()) {
    return std::unique_ptr<BoxingLogger>(
        new CsvBoxingLogger(StrCat("boxing/log/", GlobalJobDesc().job_id()) + ".csv"));
  } else {
    return std::unique_ptr<BoxingLogger>(new NullBoxingLogger());
  }
}

Maybe<void> MakeGetterTaskNode4MachineId7ThrdId(
    const std::vector<CompTaskNode*>& task_nodes,
    std::function<Maybe<CompTaskNode*>(int64_t mchn_id, int64_t thrd_id)>* Getter) {
  // ticks are shared within a machine/process
  auto machine_id2task_node = std::make_shared<HashMap<int64_t, CompTaskNode*>>();
  for (auto* task_node : task_nodes) {
    machine_id2task_node->emplace(task_node->machine_id(), task_node);
  }
  *Getter = [machine_id2task_node](int64_t mchn_id, int64_t thrd_id) -> Maybe<CompTaskNode*> {
    const auto& iter = machine_id2task_node->find(mchn_id);
    CHECK_OR_RETURN(iter != machine_id2task_node->end());
    return iter->second;
  };
  return Maybe<void>::Ok();
}

void GenSortedCompTaskNodes(const OpNode* op_node, std::vector<CompTaskNode*>* sorted_comp_tasks) {
  int64_t parallel_idx = 0;
  const ParallelDesc& parallel_desc = op_node->parallel_desc();
  int64_t parallel_num = parallel_desc.parallel_num();
  for (int64_t machine_id : parallel_desc.sorted_machine_ids()) {
    for (int64_t dev_phy_id : parallel_desc.sorted_dev_phy_ids(machine_id)) {
      CompTaskNode* comp_task_node = NewCompTaskNode4OpNode(op_node);
      comp_task_node->set_machine_id(machine_id);
      comp_task_node->mut_parallel_ctx()->set_parallel_id(parallel_idx++);
      comp_task_node->mut_parallel_ctx()->set_parallel_num(parallel_num);

      DeviceId::device_index_t device_index =
          parallel_desc.device_type() == DeviceType::kCPU
              ? DeviceId::kCPUDeviceIndex
              : static_cast<DeviceId::device_index_t>(dev_phy_id);
      DeviceId device_id{static_cast<DeviceId::rank_t>(machine_id), parallel_desc.device_type(),
                         device_index};
      StreamId::stream_index_t stream_index =
          StreamIndexGetterRegistryManager::Get().StreamIndex4DeviceIdAndTaskType(
              device_id, comp_task_node->GetTaskType());
      comp_task_node->set_thrd_id(SerializeStreamIdToInt64(StreamId{device_id, stream_index}));
      comp_task_node->set_op_node(op_node);
      sorted_comp_tasks->push_back(comp_task_node);
    }
  }
}

bool IsConnectedLbisAllSameParallelDistribution(const OpEdge* op_edge) {
  const OpNode* src_node = op_edge->src_node();
  const OpNode* dst_node = op_edge->dst_node();
  CHECK_GT(op_edge->lbis().size(), 0);
  HashSet<bool> predicators;
  for (const LogicalBlobId& lbi : op_edge->lbis()) {
    const ParallelDistribution& src_parallel_distribution = src_node->ParallelDistribution4Lbi(lbi);
    const ParallelDistribution& dst_parallel_distribution = dst_node->ParallelDistribution4Lbi(lbi);
    predicators.insert(src_parallel_distribution == dst_parallel_distribution);
  }
  CHECK_EQ(predicators.size(), 1);
  return *predicators.begin();
}

BldSubTskGphMthd GetMthdForBldSubTskGph(const OpEdge* op_edge) {
  const OpNode* src_node = op_edge->src_node();
  const OpNode* dst_node = op_edge->dst_node();
  const ParallelDesc& src_pd = src_node->parallel_desc();
  const ParallelDesc& dst_pd = dst_node->parallel_desc();
  const OperatorConf& src_op_conf = src_node->op().op_conf();
  const OperatorConf& dst_op_conf = dst_node->op().op_conf();

  // WaitAndSendIds -> Reentrantlock
  if (src_op_conf.has_wait_and_send_ids_conf() && dst_op_conf.has_reentrant_lock_conf()) {
    CHECK_EQ(src_pd.parallel_num(), 1);
    CHECK_EQ(dst_pd.parallel_num(), 1);
    return &TaskGraph::BldSubTskGphByBoxing;
  }

  // *Tick -> *Tick
  if (IsTickOpConf(src_op_conf) || IsTickOpConf(dst_op_conf)) {
    if (src_op_conf.has_source_tick_conf()) {
      CHECK(dst_op_conf.has_tick_conf());
      CHECK_EQ(src_pd.parallel_num(), 1);
      CHECK_EQ(dst_pd.parallel_num(), 1);
      return &TaskGraph::BldSubTskGphByBoxing;
    } else if (dst_op_conf.has_sink_tick_conf()) {
      CHECK(src_op_conf.has_tick_conf() || src_op_conf.has_sink_tick_conf());
      CHECK_EQ(src_pd.parallel_num(), 1);
      CHECK_EQ(dst_pd.parallel_num(), 1);
      return &TaskGraph::BldSubTskGphByBoxing;
    } else if (IsSubsetTickOpConf(src_op_conf)) {
      return &TaskGraph::BldSubTskGphBySrcSubsetConnect;
    } else if (IsSubsetTickOpConf(dst_op_conf)) {
      return &TaskGraph::BldSubTskGphByDstSubsetConnect;
    } else if (IsTickOpConf(src_op_conf) && IsTickOpConf(dst_op_conf)) {
      if (src_pd.parallel_num() == dst_pd.parallel_num()) {
        return &TaskGraph::BldSubTskGphByOneToOne;
      } else {
        CHECK_EQ(src_pd.parallel_num(), 1);
        return &TaskGraph::BldSubTskGphByBroadcastToBroadcast;
      }
    }
  }

  std::shared_ptr<CompTaskNode> src_comp_task(NewCompTaskNode4OpNode(src_node));
  std::shared_ptr<CompTaskNode> dst_comp_task(NewCompTaskNode4OpNode(dst_node));
  // NOTE(chengcheng): MUST use TaskType instead of OpTypeCase because may
  //   Multi-op correspoding to SAME TaskType such as:
  //     DistributeConcatOpConf and DistributeAddOpConf -> TaskType::kDistributeConcat
  //     DistributeSplitOpConf  and DistributeCloneOpConf -> TaskType::kDistributeSplit
  // * -> DistributeConcat
  if (dst_comp_task->GetTaskType() == TaskType::kDistributeConcat) {
    return &TaskGraph::BldSubTskGphByPartialInLbiConnect;
  }

  // DistributeSplit -> *
  if (src_comp_task->GetTaskType() == TaskType::kDistributeSplit) {
    return &TaskGraph::BldSubTskGphByPartialOutLbiConnect;
  }

  // NormalForward -> DecodeH2D
  if (src_comp_task->GetTaskType() == TaskType::kNormalForward
      && dst_comp_task->GetTaskType() == TaskType::kDecodeH2D) {
    return &TaskGraph::BldSubTskGphNormalForwardToDecodeH2D;
  }

  if (src_pd.parallel_num() == 1 && dst_pd.parallel_num() == 1) {
    return &TaskGraph::BldSubTskGphByOneToOne;
  }

  // one to one
  if (src_pd.parallel_num() == dst_pd.parallel_num() && *src_pd.hierarchy() == *dst_pd.hierarchy()
      && IsConnectedLbisAllSameParallelDistribution(op_edge)) {
    return &TaskGraph::BldSubTskGphByOneToOne;
  }

  return &TaskGraph::BldSubTskGphByBoxing;
}

void ForEachOpGraphNecessaryCtrlEdge(
    const OpGraph* op_graph,
    const std::function<void(const OpNode*, const OpNode*, int64_t)>& Handler) {
  auto IsOpGraphDataReachable = op_graph->MakePredicatorIsReachable();
  op_graph->ForEachNode([&](OpNode* dst) {
    for (const auto& ctrl_in_op_name : dst->op().op_conf().ctrl_in_op_name()) {
      const OpNode* src = op_graph->OpNode4OpName(ctrl_in_op_name);
      CHECK(!IsOpGraphDataReachable(dst, src));
      if (!IsOpGraphDataReachable(src, dst)) {
        CHECK(src->parallel_desc().EqualsIgnoringDeviceType(dst->parallel_desc()));
        const Shape* src_time_shape = CHECK_JUST(src->op().GetOpTimeShape()).get();
        const Shape* dst_time_shape = CHECK_JUST(dst->op().GetInputBlobFastestTimeShape()).get();
        if (dst_time_shape == nullptr) {
          dst_time_shape = CHECK_JUST(dst->op().GetOpTimeShape()).get();
        }
        CHECK(src_time_shape->elem_cnt() == dst_time_shape->elem_cnt()
              || src_time_shape->Containing(*dst_time_shape));
        CHECK_EQ(src_time_shape->elem_cnt() % dst_time_shape->elem_cnt(), 0);
        int64_t regst_desc_num = src_time_shape->elem_cnt() / dst_time_shape->elem_cnt();
        Handler(src, dst, regst_desc_num);
      }
    }
  });
}

}  // namespace

TaskGraph::TaskGraph() {
  OpGraph* op_graph = Global<OpGraph>::Get();
  sub_tsk_gph_builder_ctx_.reset(new SubTskGphBuilderCtx(this));
  boxing_logger_ = CreateBoxingLogger();
  hierarchical_sub_tsk_gph_builder_.reset(new DispatchHierarchicalSubTskGphBuilder());
  HashMap<const OpNode*, std::vector<CompTaskNode*>> op_node2sorted_comp_tasks;

  op_graph->ForEachNode([&](const OpNode* op_node) {
    std::vector<CompTaskNode*>* sorted_comp_tasks = &(op_node2sorted_comp_tasks[op_node]);
    GenSortedCompTaskNodes(op_node, sorted_comp_tasks);
    for (CompTaskNode* comp_task : *sorted_comp_tasks) { AddAllocatedNode(comp_task); }
  });

  op_graph->ForEachEdge([&](const OpEdge* op_edge) {
    BldSubTskGphMthd method = GetMthdForBldSubTskGph(op_edge);
    (this->*method)(op_edge, op_node2sorted_comp_tasks.at(op_edge->src_node()),
                    op_node2sorted_comp_tasks.at(op_edge->dst_node()));
  });

  ForEachOpGraphNecessaryCtrlEdge(
      op_graph, [&](const OpNode* src, const OpNode* dst, int64_t ctrl_regst_num) {
        const auto& src_task_nodes = op_node2sorted_comp_tasks.at(src);
        const auto& dst_task_nodes = op_node2sorted_comp_tasks.at(dst);
        if (src->op().op_conf().has_src_subset_tick_conf()) {
          UNIMPLEMENTED();
        } else if (dst->op().op_conf().has_dst_subset_tick_conf()) {
          UNIMPLEMENTED();
        } else {
          ConnectCtrlEdges(src_task_nodes, dst_task_nodes, ctrl_regst_num);
        }
      });

  SetOrderInGraphForEachNode();
  if (Global<ResourceDesc, ForSession>::Get()->enable_debug_mode()) { ToDotWithAutoFilePath(); }
}

TaskGraph::~TaskGraph() = default;

TaskEdge* TaskGraph::NewTaskEdgeWithLbi(const LogicalBlobId& lbi) {
  TaskEdge* edge = NewEdge();
  edge->AddLbi(lbi);
  return edge;
}

TaskEdge* TaskGraph::NewTaskEdgeWithLbis(const std::vector<LogicalBlobId>& lbis) {
  TaskEdge* edge = NewEdge();
  edge->AddLbis(lbis);
  return edge;
}

TaskNode* TaskGraph::GetProxyNode(TaskNode* src_node, const LogicalBlobId& lbi,
                                  int64_t dst_machine_id, int64_t dst_mem_zone_id) {
  int64_t src_mem_zone_id = src_node->MemZoneId121();
  const ProxyKey key(lbi, dst_machine_id, dst_mem_zone_id);
  if (node2proxies_.find(src_node) != node2proxies_.cend()
      && node2proxies_.at(src_node).find(key) != node2proxies_.at(src_node).cend()) {
    return node2proxies_.at(src_node).at(key);
  } else {
    if (dst_machine_id == src_node->machine_id() && dst_mem_zone_id == src_mem_zone_id) {
      node2proxies_[src_node][key] = src_node;
      return src_node;
    } else if (Global<IDMgr>::Get()->IsGpuMemZone(dst_mem_zone_id)) {
      TaskNode* proxy_on_dst_host =
          GetProxyNode(src_node, lbi, dst_machine_id, Global<IDMgr>::Get()->CpuMemZoneId());
      CopyHdTaskNode* copy_task = NewNode<CopyHdTaskNode>();
      copy_task->Init(CopyHdOpConf::H2D, proxy_on_dst_host->machine_id(),
                      Global<IDMgr>::Get()->GetGpuPhyIdFromMemZoneId(dst_mem_zone_id), lbi);
      Connect<TaskNode>(proxy_on_dst_host, NewTaskEdgeWithLbi(lbi), copy_task);
      node2proxies_[src_node][key] = copy_task;
      return copy_task;
    } else if (Global<IDMgr>::Get()->IsCpuMemZone(dst_mem_zone_id)) {
      if (src_node->machine_id() == dst_machine_id) {
        if (Global<IDMgr>::Get()->IsGpuMemZone(src_mem_zone_id)) {
          CopyHdTaskNode* copy_task = NewNode<CopyHdTaskNode>();
          copy_task->Init(CopyHdOpConf::D2H, src_node->machine_id(),
                          Global<IDMgr>::Get()->GetGpuPhyIdFromMemZoneId(src_mem_zone_id), lbi);
          Connect<TaskNode>(src_node, NewTaskEdgeWithLbi(lbi), copy_task);
          node2proxies_[src_node][key] = copy_task;
          return copy_task;
        } else {
          UNIMPLEMENTED();
        }
      } else {
        TaskNode* proxy_on_src_host = GetProxyNode(src_node, lbi, src_node->machine_id(),
                                                   Global<IDMgr>::Get()->CpuMemZoneId());
        CopyCommNetTaskNode* copy_comm_net_task = NewNode<CopyCommNetTaskNode>();
        copy_comm_net_task->Init(dst_machine_id, lbi);
        Connect<TaskNode>(proxy_on_src_host, NewTaskEdgeWithLbi(lbi), copy_comm_net_task);
        node2proxies_[src_node][key] = copy_comm_net_task;
        return copy_comm_net_task;
      }
    } else {
      UNIMPLEMENTED();
    }
  }
  UNIMPLEMENTED();
  return nullptr;
}

TaskNode* TaskGraph::GetProxyNode(TaskNode* src_node, const LogicalBlobId& lbi,
                                  const ParallelDesc& dst_parallel_desc, int64_t dst_parallel_id) {
  const int64_t dst_machine_id =
      CHECK_JUST(dst_parallel_desc.MachineId4ParallelId(dst_parallel_id));
  int64_t dst_mem_zone_id;
  const IDMgr* id_mgr = Global<IDMgr>::Get();
  if (dst_parallel_desc.device_type() == DeviceType::kCPU) {
    dst_mem_zone_id = id_mgr->CpuMemZoneId();
  } else if (dst_parallel_desc.device_type() == DeviceType::kGPU) {
    const int64_t dst_dev_phy_id =
        CHECK_JUST(dst_parallel_desc.DeviceId4ParallelId(dst_parallel_id));
    dst_mem_zone_id = id_mgr->GpuMemZoneId(dst_dev_phy_id);
  } else {
    UNIMPLEMENTED();
  }
  return GetProxyNode(src_node, lbi, dst_machine_id, dst_mem_zone_id);
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

void TaskGraph::RemoveEmptyRegsts() {
  ForEachNode([&](TaskNode* node) { node->EraseZeroSizeProducedBlob(); });
  ForEachNode([&](TaskNode* node) { node->EraseZeroSizeConsumedRegst(); });
  ForEachNode([&](TaskNode* node) { node->EraseZeroSizeProducedRegst(); });
  ForEachNode([&](TaskNode* node) { node->UnbindBnWithEmptyRegst(); });
}

void TaskGraph::MergeChainAndAddOrderingCtrlEdgeInSameChain() {
  MergeChain();
  BuildCtrlRegstDescInSameChain();
}

void TaskGraph::SetOrderInGraphForEachNode() {
  int64_t order_in_graph = 0;
  auto SetOrderInGraph = [&](TaskNode* task_node) {
    task_node->set_order_in_graph(order_in_graph);
    ordered_task_nodes_.emplace_back(task_node);
    ++order_in_graph;
  };
  TopoForEachNode(SetOrderInGraph);
}

void TaskGraph::MergeChain() {
  int64_t chain_id = 0;
  for (auto* this_node : ordered_task_nodes_) {
    // skip if this node has been set in a chain.
    if (this_node->chain_id() != -1) { continue; }

    CHECK_EQ(this_node->chain_id(), -1);
    if (CanBeMergedInChain(this_node)) {
      TraverseConnectedSubGraphMergeInThisChain(this_node, chain_id);
    } else {
      this_node->set_chain_id(chain_id);
    }

    ++chain_id;
  }
  for (auto* node : ordered_task_nodes_) { CHECK_NE(node->chain_id(), -1); }
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
      TaskNode* src_node = iter->second;
      TaskNode* dst_node = node;
      std::string ctrl_regst_name;
      bool build_ctrl_edge = src_node->BuildCtrlRegstDescIfNeed(dst_node, &ctrl_regst_name);
      if (build_ctrl_edge) {
        CHECK(!ctrl_regst_name.empty());
        TaskEdge* edge = NewEdge();
        Connect<TaskNode>(src_node, edge, dst_node);
        src_node->BindEdgeWithProducedRegst(edge, ctrl_regst_name);
      }
      iter->second = dst_node;
    }
  }
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
    for (const auto& pair : task_node->exec_gph().SoleNode()->mut_inplace_obn2ibn()) {
      AddMutableInplaceArgPair(task_node, pair.second, pair.first, op.op_name());
    }
    for (const auto& pair : task_node->exec_gph().SoleNode()->con_inplace_obn2ibn()) {
      AddConstInplaceArgPair(task_node, pair.second, pair.first, op.op_name());
    }
  }
}

void TaskGraph::GetSafeInplaceOpBlobArgList(
    InplaceObasInfo* safe_obas_info, const HashSet<TaskNode*>& dev_nodes,
    const std::function<bool(const std::string&, const std::string&)>& IsOpNameDataOrCtrlReachable)
    const {
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
  if (Global<ResourceDesc, ForSession>::Get()->enable_debug_mode()) {
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
  inplace_gph.ForEachConnectedComponent([&](const HashSet<const InplaceLbiNode*>& inplace_nodes) {
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

#define DEFINE_BLD_SUB_TASK_GRAPH_METHOD(method_name) \
  void TaskGraph::method_name BLD_SUB_TSK_GPH_MTHD_ARGS()

DEFINE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphByBoxing) {
  const OpNode* src_op_node = op_edge->src_node();
  const OpNode* dst_op_node = op_edge->dst_node();
  for (const LogicalBlobId& lbi : op_edge->lbis()) {
    std::vector<TaskNode*> in_nodes(sorted_src_comp_tasks.begin(), sorted_src_comp_tasks.end());
    std::vector<TaskNode*> out_nodes;
    out_nodes.reserve(sorted_dst_comp_tasks.size());
    std::vector<std::vector<TaskNode*>> sorted_ctrl_tasks;
    const ParallelDistribution& src_parallel_distribution =
        src_op_node->ParallelDistribution4Lbi(lbi);
    const ParallelDistribution& dst_parallel_distribution =
        dst_op_node->ParallelDistribution4Lbi(lbi);
    const ParallelDesc& src_parallel_desc = src_op_node->parallel_desc();
    const ParallelDesc& dst_parallel_desc = dst_op_node->parallel_desc();
    const BlobDesc& blob_desc = src_op_node->LogicalBlobDesc4Lbi(lbi);
    auto status = CHECK_JUST(hierarchical_sub_tsk_gph_builder_->Build(
        sub_tsk_gph_builder_ctx_.get(), in_nodes, &out_nodes, &sorted_ctrl_tasks, src_parallel_desc,
        dst_parallel_desc, lbi, blob_desc, src_parallel_distribution, dst_parallel_distribution,
        *(CHECK_JUST(src_op_node->op().GetOpTimeShape()).get())));
    boxing_logger_->Log(*status, src_op_node->op().op_name(), dst_op_node->op().op_name(),
                        src_parallel_desc, dst_parallel_desc, src_parallel_distribution,
                        dst_parallel_distribution, lbi, blob_desc);
    CHECK_EQ(out_nodes.size(), sorted_dst_comp_tasks.size());
    FOR_RANGE(size_t, i, 0, out_nodes.size()) {
      ConnectWithLbi(out_nodes.at(i), sorted_dst_comp_tasks.at(i), lbi);
    }
    if (!sorted_ctrl_tasks.empty()) {
      CHECK_EQ(sorted_ctrl_tasks.size(), sorted_dst_comp_tasks.size());
      FOR_RANGE(size_t, i, 0, sorted_dst_comp_tasks.size()) {
        for (TaskNode* ctrl_node : sorted_ctrl_tasks.at(i)) {
          std::string regst_desc_name;
          ctrl_node->BuildCtrlRegstDesc(sorted_dst_comp_tasks.at(i), &regst_desc_name);
          TaskEdge* edge = NewEdge();
          Connect<TaskNode>(ctrl_node, edge, sorted_dst_comp_tasks.at(i));
          ctrl_node->BindEdgeWithProducedRegst(edge, regst_desc_name);
        }
      }
    }
  }
}

DEFINE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphByOneToOne) {
  CHECK_EQ(sorted_src_comp_tasks.size(), sorted_dst_comp_tasks.size());
  FOR_RANGE(size_t, i, 0, sorted_src_comp_tasks.size()) {
    for (const LogicalBlobId& lbi : op_edge->lbis()) {
      BuildTaskPath(sorted_src_comp_tasks.at(i), sorted_dst_comp_tasks.at(i), lbi);
    }
  }
}

DEFINE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphByBroadcastToBroadcast) {
  for (CompTaskNode* dst_node : sorted_dst_comp_tasks) {
    CompTaskNode* nearest_src_node =
        SubTskGphBuilderUtil::FindNearestNode(sorted_src_comp_tasks, dst_node);
    CHECK_NOTNULL(nearest_src_node);
    for (const LogicalBlobId& lbi : op_edge->lbis()) {
      BuildTaskPath(nearest_src_node, dst_node, lbi);
    }
  }
}

DEFINE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphByPartialInLbiConnect) {
  const Operator& src_op = op_edge->src_node()->op();
  const Operator& dst_op = op_edge->dst_node()->op();
  HashSet<LogicalBlobId> lbis;
  for (const auto& obn : src_op.output_bns()) { lbis.insert(src_op.BnInOp2Lbi(obn)); }
  CHECK_EQ(sorted_src_comp_tasks.size(), 1);
  CHECK_EQ(dst_op.input_bns().size(), sorted_dst_comp_tasks.size());
  FOR_RANGE(int, i, 0, sorted_dst_comp_tasks.size()) {
    const auto& lbi = dst_op.BnInOp2Lbi(dst_op.input_bns().Get(i));
    if (lbis.find(lbi) != lbis.end()) {
      BuildTaskPath(sorted_src_comp_tasks.at(0), sorted_dst_comp_tasks.at(i), lbi);
    }
  }
}

DEFINE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphByPartialOutLbiConnect) {
  const Operator& src_op = op_edge->src_node()->op();
  const Operator& dst_op = op_edge->dst_node()->op();
  HashSet<LogicalBlobId> lbis;
  for (const auto& ibn : dst_op.input_bns()) { lbis.insert(dst_op.BnInOp2Lbi(ibn)); }
  CHECK_EQ(sorted_dst_comp_tasks.size(), 1);
  CHECK_EQ(src_op.output_bns().size(), sorted_src_comp_tasks.size());
  FOR_RANGE(int, i, 0, sorted_src_comp_tasks.size()) {
    const auto& lbi = src_op.BnInOp2Lbi(src_op.output_bns().Get(i));
    if (lbis.find(lbi) != lbis.end()) {
      BuildTaskPath(sorted_src_comp_tasks.at(i), sorted_dst_comp_tasks.at(0), lbi);
    }
  }
}

DEFINE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphBySrcSubsetConnect) {
  std::function<Maybe<CompTaskNode*>(int64_t mchn_id, int64_t thrd_id)> TaskNode4MachineId7ThrdId;
  CHECK_JUST(
      MakeGetterTaskNode4MachineId7ThrdId(sorted_src_comp_tasks, &TaskNode4MachineId7ThrdId));
  for (CompTaskNode* dst_task_node : sorted_dst_comp_tasks) {
    CompTaskNode* src_task_node = CHECK_JUST(
        TaskNode4MachineId7ThrdId(dst_task_node->machine_id(), dst_task_node->thrd_id()));
    Connect<TaskNode>(src_task_node, NewTaskEdgeWithLbis(op_edge->lbis()), dst_task_node);
  }
}

DEFINE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphByDstSubsetConnect) {
  std::function<Maybe<CompTaskNode*>(int64_t mchn_id, int64_t thrd_id)> TaskNode4MachineId7ThrdId;
  CHECK_JUST(
      MakeGetterTaskNode4MachineId7ThrdId(sorted_dst_comp_tasks, &TaskNode4MachineId7ThrdId));
  for (CompTaskNode* src_task_node : sorted_src_comp_tasks) {
    CompTaskNode* dst_task_node = CHECK_JUST(
        TaskNode4MachineId7ThrdId(src_task_node->machine_id(), src_task_node->thrd_id()));
    Connect<TaskNode>(src_task_node, NewTaskEdgeWithLbis(op_edge->lbis()), dst_task_node);
  }
}

DEFINE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphNormalForwardToDecodeH2D) {
  CHECK_EQ(sorted_src_comp_tasks.size(), sorted_dst_comp_tasks.size());
  FOR_RANGE(size_t, i, 0, sorted_src_comp_tasks.size()) {
    CompTaskNode* src = sorted_src_comp_tasks.at(i);
    CompTaskNode* dst = sorted_dst_comp_tasks.at(i);
    for (const LogicalBlobId& lbi : op_edge->lbis()) { BuildTaskPath(src, dst, lbi); }
  }
}

void TaskGraph::ConnectWithLbi(TaskNode* src_node, TaskNode* dst_node, const LogicalBlobId& lbi) {
  if (src_node == dst_node) { return; }
  for (TaskEdge* out_edge : src_node->out_edges()) {
    TaskNode* out_node = out_edge->dst_node();
    if (out_node == dst_node) {
      out_edge->AddLbi(lbi);
      return;
    }
  }

  TaskEdge* connected_edge = NewEdge();
  connected_edge->AddLbi(lbi);
  Connect<TaskNode>(src_node, connected_edge, dst_node);
}

void TaskGraph::BuildTaskPath(TaskNode* src_node, TaskNode* dst_node, const LogicalBlobId& lbi) {
  int64_t dst_machine_id = dst_node->machine_id();
  int64_t dst_mem_zone_id = dst_node->MemZoneId121();
  TaskNode* proxy_node = GetProxyNode(src_node, lbi, dst_machine_id, dst_mem_zone_id);
  ConnectWithLbi(proxy_node, dst_node, lbi);
}

}  // namespace oneflow
