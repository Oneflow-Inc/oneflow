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
#include "oneflow/core/common/global.h"
#include "oneflow/core/common/hash_container.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/graph/inplace_lbi_graph.h"
#include "oneflow/core/graph/task_node.h"
#include "oneflow/core/job/task.pb.h"
#include "oneflow/core/register/blob_desc.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/operator/variable_op.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/graph/normal_forward_compute_task_node.h"
#include "oneflow/core/graph/boxing_identity_task_node.h"
#include "oneflow/core/job/scope.h"
#include "oneflow/core/register/logical_blob_id.pb.h"
#include "oneflow/core/register/register_desc.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"
#include "oneflow/core/vm/symbol_storage.h"
#include "oneflow/core/job_rewriter/calculation_pass.h"
#include "oneflow/core/graph/boxing/sub_task_graph_builder_util.h"
#include "oneflow/core/graph/boxing/hierarchical_sub_task_graph_builder_impl.h"
#include "oneflow/core/graph/task_stream_index_manager.h"
#include "oneflow/core/ep/include/primitive/memcpy.h"
#include "oneflow/core/job_rewriter/straighten_nodes.h"

namespace oneflow {

namespace {

bool IsMemcpyPrimitiveSupported(DeviceType device_type, ep::primitive::MemcpyKind kind) {
  auto primitive = ep::primitive::NewPrimitive<ep::primitive::MemcpyFactory>(device_type, kind);
  return primitive.operator bool();
}

bool IsMemcpyHtoDSupported(DeviceType device_type) {
  return IsMemcpyPrimitiveSupported(device_type, ep::primitive::MemcpyKind::kHtoD);
}

bool IsMemcpyDtoHSupported(DeviceType device_type) {
  return IsMemcpyPrimitiveSupported(device_type, ep::primitive::MemcpyKind::kDtoH);
}

bool IsConnectToTickOp(const TaskNode* node) {
  const auto* comp_task_node = dynamic_cast<const CompTaskNode*>(node);
  if (comp_task_node == nullptr) { return false; }
  const Operator* op = comp_task_node->op().get();
  if (dynamic_cast<const VariableOp*>(op) != nullptr) { return true; }
  return false;
}

std::string GetOpConfCalculationPassName(const OperatorConf& op_conf) {
  CHECK(op_conf.has_scope_symbol_id());
  int64_t scope_symbol_id = op_conf.scope_symbol_id();
  CHECK(Global<symbol::Storage<Scope>>::Get()->Has(scope_symbol_id))
      << " Error! op : \n " << op_conf.DebugString()
      << " has error scope_symbol_id = " << scope_symbol_id
      << " which cannot find in Global<symbol::Storage<Scope>>::Get()\n";
  const Scope& scope = Global<symbol::Storage<Scope>>::Get()->Get(scope_symbol_id);
  return scope.scope_proto().calculation_pass_name();
}

bool IsOptimizerPassOp(const Operator* op) {
  // NOTE(chengcheng): use scope::calculation_pass_name instead of area_id to not merge optimizer
  // ops with fw/bw ops
  if (!op->op_conf().has_scope_symbol_id()) {
    // NOTE(chengcheng): Some system op insert to OpGraph may not set scope_symbol_id, it MUST NOT
    // optimizer subgraph ops.
    return false;
  }
  return GetOpConfCalculationPassName(op->op_conf()) == kOptimizerPass;
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
  if (op_conf.has_user_conf()) {
    const std::string& user_type_name = op_conf.user_conf().op_type_name();
    if (user_type_name == "repeat" || user_type_name == "acc" || user_type_name == "pack"
        || user_type_name == "unpack" || user_type_name == "identity_buffer") {
      return true;
    }
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
  if (fw_comp_node->device_type() != DeviceType::kCUDA) { return false; }
  const Operator* op = fw_comp_node->op().get();
  if (IsSpecialOpNotConsiderMergeInChain(op)) { return false; }
  return true;
}

std::shared_ptr<const Shape> GetTaskNodeTimeShape(const TaskNode* node) {
  const auto* fw_comp_node = dynamic_cast<const NormalForwardCompTaskNode*>(node);
  CHECK(fw_comp_node != nullptr);
  return CHECK_JUST(fw_comp_node->op()->GetOpTimeShape());
}

void TraverseConnectedSubGraphMergeInThisChain(TaskNode* this_node, const int64_t this_chain_id) {
  CHECK_NE(this_chain_id, -1);
  CHECK_EQ(this_node->chain_id(), -1);
  // bfs search all node can be merged in this chain
  std::shared_ptr<const Shape> seed_time_shape = GetTaskNodeTimeShape(this_node);
  HashSet<TaskNode*> visited_nodes;
  std::queue<TaskNode*> queued_nodes;
  queued_nodes.push(this_node);
  visited_nodes.insert(this_node);
  while (!queued_nodes.empty()) {
    TaskNode* cur_node = queued_nodes.front();
    queued_nodes.pop();

    CHECK_EQ(cur_node->chain_id(), -1);
    cur_node->set_chain_id(this_chain_id);

    cur_node->ForEachNodeOnInOutDataEdge([&](TaskNode* next_node) {
      if (visited_nodes.find(next_node) == visited_nodes.end() && CanBeMergedInChain(next_node)
          && this_node->thrd_id() == next_node->thrd_id()
          && (*GetTaskNodeTimeShape(next_node)) == (*seed_time_shape)) {
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
              ? 0
              : static_cast<DeviceId::device_index_t>(dev_phy_id);
      DeviceId device_id{static_cast<DeviceId::rank_t>(machine_id), parallel_desc.device_type(),
                         device_index};
      StreamId::stream_index_t stream_index = 0;
      if (op_node->op().op_conf().has_stream_name_hint()) {
        const std::string& stream_name_hint = op_node->op().op_conf().stream_name_hint();
        VLOG(3) << "set op: " << op_node->op().op_name() << " to stream: " << stream_name_hint;
        stream_index = Global<TaskStreamIndexManager>::Get()->GetNamedTaskStreamIndex(
            device_id, stream_name_hint);
      } else {
        stream_index = Global<TaskStreamIndexManager>::Get()->GetTaskStreamIndex(
            comp_task_node->GetTaskType(), device_id);
      }
      comp_task_node->set_thrd_id(EncodeStreamIdToInt64(StreamId{device_id, stream_index}));
      comp_task_node->set_op_node(op_node);
      sorted_comp_tasks->emplace_back(comp_task_node);
    }
  }
}

bool IsConnectedLbisAllSameNdSbp(const OpEdge* op_edge) {
  const OpNode* src_node = op_edge->src_node();
  const OpNode* dst_node = op_edge->dst_node();
  CHECK_GT(op_edge->lbis().size(), 0);
  HashSet<bool> predicators;
  for (const LogicalBlobId& lbi : op_edge->lbis()) {
    const NdSbp& src_nd_sbp = src_node->NdSbp4Lbi(lbi);
    const NdSbp& dst_nd_sbp = dst_node->NdSbp4Lbi(lbi);
    predicators.insert(src_nd_sbp == dst_nd_sbp);
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
      && IsConnectedLbisAllSameNdSbp(op_edge)) {
    return &TaskGraph::BldSubTskGphByOneToOne;
  }

  return &TaskGraph::BldSubTskGphByBoxing;
}

void ForEachOpGraphNecessaryCtrlEdge(
    const OpGraph* op_graph, const std::function<void(const OpNode*, const OpNode*)>& Handler) {
  auto IsOpGraphDataReachable = op_graph->MakePredicatorIsReachable();
  op_graph->ForEachNode([&](OpNode* dst) {
    for (const auto& ctrl_in_op_name : dst->op().op_conf().ctrl_in_op_name()) {
      const OpNode* src = op_graph->OpNode4OpName(ctrl_in_op_name);
      CHECK(!IsOpGraphDataReachable(dst, src));
      if (!IsOpGraphDataReachable(src, dst)) {
        CHECK_EQ(dst->parallel_desc().parallel_num(), src->parallel_desc().parallel_num());
        const Shape* src_time_shape = CHECK_JUST(src->op().GetOpTimeShape()).get();
        const Shape* dst_time_shape = CHECK_JUST(dst->op().GetInputBlobFastestTimeShape()).get();
        if (dst_time_shape == nullptr) {
          dst_time_shape = CHECK_JUST(dst->op().GetOpTimeShape()).get();
        }
        CHECK_EQ(src_time_shape->elem_cnt(), dst_time_shape->elem_cnt());
        Handler(src, dst);
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

  ForEachOpGraphNecessaryCtrlEdge(op_graph, [&](const OpNode* src, const OpNode* dst) {
    const auto& src_task_nodes = op_node2sorted_comp_tasks.at(src);
    const auto& dst_task_nodes = op_node2sorted_comp_tasks.at(dst);
    if (src->op().op_conf().has_src_subset_tick_conf()) {
      UNIMPLEMENTED();
    } else if (dst->op().op_conf().has_dst_subset_tick_conf()) {
      UNIMPLEMENTED();
    } else {
      ConnectCtrlEdges(src_task_nodes, dst_task_nodes);
    }
  });
  if (ParseBooleanFromEnv("ONEFLOW_RANDOM_STRAIGHTEN_NODES", false)) {
    SetOrderInGraphForEachNode();
  } else {
    StraightenNodes();
  }

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
                                  const MemZoneId& dst_mem_zone_id) {
  const auto& src_mem_zone_id = src_node->MemZoneId121();
  const ProxyKey key(src_node, lbi, dst_mem_zone_id);
  auto it = proxy2node.find(key);
  if (it != proxy2node.cend()) {
    // hit cache
    return it->second;
  } else {
    if (src_mem_zone_id == dst_mem_zone_id) {
      // in the same memory zone
      proxy2node[key] = src_node;
      return src_node;
    } else if (dst_mem_zone_id.device_type() == DeviceType::kCPU) {
      if (src_mem_zone_id.rank() == dst_mem_zone_id.rank()) {
        // on the same node, not on the same device
        // src must be not on the cpu mem zone, copy d2h first
        CHECK(IsMemcpyDtoHSupported(src_mem_zone_id.device_type()));
        CopyHdTaskNode* copy_task = NewNode<CopyHdTaskNode>();
        copy_task->Init(CopyHdOpConf::D2H, src_mem_zone_id, lbi);
        Connect<TaskNode>(src_node, NewTaskEdgeWithLbi(lbi), copy_task);
        proxy2node[key] = copy_task;
        return copy_task;
      } else {
        // not on the same node, need CopyCommNet from src to dst
        // build src cpu proxy first
        TaskNode* proxy_on_src_host =
            GetProxyNode(src_node, lbi, GetNodeCPUMemZoneId(src_mem_zone_id.rank()));
        CopyCommNetTaskNode* copy_comm_net_task = NewNode<CopyCommNetTaskNode>();
        copy_comm_net_task->Init(dst_mem_zone_id.rank(), lbi);
        Connect<TaskNode>(proxy_on_src_host, NewTaskEdgeWithLbi(lbi), copy_comm_net_task);
        proxy2node[key] = copy_comm_net_task;
        return copy_comm_net_task;
      }
    } else {
      TaskNode* proxy_on_dst_host =
          GetProxyNode(src_node, lbi, GetNodeCPUMemZoneId(dst_mem_zone_id.rank()));
      CHECK(IsMemcpyHtoDSupported(dst_mem_zone_id.device_type()));
      CopyHdTaskNode* copy_task = NewNode<CopyHdTaskNode>();
      copy_task->Init(CopyHdOpConf::H2D, dst_mem_zone_id, lbi);
      Connect<TaskNode>(proxy_on_dst_host, NewTaskEdgeWithLbi(lbi), copy_task);
      proxy2node[key] = copy_task;
      return copy_task;
    }
  }
  return nullptr;
}

TaskNode* TaskGraph::GetProxyNode(TaskNode* src_node, const LogicalBlobId& lbi,
                                  const ParallelDesc& dst_parallel_desc, int64_t dst_parallel_id) {
  const int64_t dst_machine_id =
      CHECK_JUST(dst_parallel_desc.MachineId4ParallelId(dst_parallel_id));
  const int64_t dev_id = CHECK_JUST(dst_parallel_desc.DeviceId4ParallelId(dst_parallel_id));
  DeviceType device_type = dst_parallel_desc.device_type();
  auto device_index =
      (device_type == DeviceType::kCPU ? 0 : static_cast<DeviceId::device_index_t>(dev_id));
  MemZoneId mem_zone_id{static_cast<MemZoneId::rank_t>(dst_machine_id), device_type, device_index};
  return GetProxyNode(src_node, lbi, mem_zone_id);
}

void TaskGraph::ConnectCtrlEdges(const std::vector<CompTaskNode*>& src_task_nodes,
                                 const std::vector<CompTaskNode*>& dst_task_nodes) {
  CHECK_EQ(src_task_nodes.size(), dst_task_nodes.size());
  FOR_RANGE(int32_t, i, 0, src_task_nodes.size()) {
    std::string regst_desc_name;
    src_task_nodes.at(i)->BuildCtrlRegstDesc(dst_task_nodes.at(i), &regst_desc_name);
    TaskEdge* edge = NewEdge();
    Connect<TaskNode>(src_task_nodes.at(i), edge, dst_task_nodes.at(i));
    src_task_nodes.at(i)->BindEdgeWithProducedRegst(edge, regst_desc_name);
  }
}

void TaskGraph::RemoveEmptyRegsts() {
  ForEachNode([&](TaskNode* node) { node->EraseUninitializedShapeProducedBlob(); });
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
  HashMap<int32_t, int32_t> task_type_map;
  auto SetOrderInGraph = [&](TaskNode* task_node) {
    if (GlobalProcessCtx::Rank() == 0) {
      std::cout << "Execution order: " << order_in_graph << ": " << task_node->VisualStr()
                << ": task type: " << task_node->GetTaskType() << ", "
                << (task_node->parallel_ctx() == 0) << std::endl;
      if (task_type_map.find(task_node->GetTaskType()) == task_type_map.end()) {
        task_type_map[task_node->GetTaskType()] = 0;
      }
      task_type_map[task_node->GetTaskType()]++;
    }
    task_node->set_order_in_graph(order_in_graph);
    ordered_task_nodes_.emplace_back(task_node);
    ++order_in_graph;
  };
  TopoForEachNodeFast(SetOrderInGraph);
  if (GlobalProcessCtx::Rank() == 0) {
    std::cout << "Print all task type: " << std::endl;
    for (auto& pair : task_type_map) {
      std::cout << "task type: " << pair.first << ", " << pair.second << std::endl;
    }
  }
}

void TaskGraph::StraightenNodes() {
  // The function for settle the order in the graph
  int64_t order_in_graph = 0;
  HashMap<int32_t, int32_t> task_type_map;

  // Generate topological data structure for each task node
  HashMap<TaskNode*, TopoStruct> task_node2topo_struct;
  TopoForEachNodeFast([&](TaskNode* node) {
    auto& topo_struct = task_node2topo_struct[node];
    topo_struct.node = node;
    if (node->in_edges().empty()) {
      topo_struct.MinLayer = 0;
    } else {
      int32_t max_min_layer = 0;
      node->ForEachNodeOnInEdge([&](TaskNode* in) {
        max_min_layer = std::max(max_min_layer, task_node2topo_struct[in].MinLayer);
      });
      topo_struct.MinLayer = max_min_layer + 1;
    }
  });

  // Generate other parameters in the topological data structure
  FindMainstem(task_node2topo_struct);

  // test debug
  if (GlobalProcessCtx::Rank() == 0) {
    std::cout << "Straightening order type: " << ParseIntegerFromEnv("Parameter0", 0) << ", "
              << ParseIntegerFromEnv("Parameter1", 1) << ", "
              << ParseIntegerFromEnv("Parameter2", 2) << std::endl;
  }
  // Order in the waiting sets
  // Decide which node should run first
  struct comp {
    bool operator()(const TopoStruct* a, const TopoStruct* b) const {
      static std::vector<int64_t> decide_parameters({ParseIntegerFromEnv("Parameter0", 0),
                                                     ParseIntegerFromEnv("Parameter1", 1),
                                                     ParseIntegerFromEnv("Parameter2", 2)});
      for (int32_t decide_parameter : decide_parameters) {
        int32_t decide_parameter_a = a->GetDecidingParameter(decide_parameter);
        int32_t decide_parameter_b = b->GetDecidingParameter(decide_parameter);
        if (decide_parameter_a != decide_parameter_b) {
          return decide_parameter_a < decide_parameter_b;
        }
      }
      return a->node->node_id() < b->node->node_id();
      // auto comp_str = a->node->VisualStr().compare(b->node->VisualStr());
      // if (comp_str == 0) {
      //   // the order does not matter right now, but we need a strict order
      //   return a < b;
      // } else {
      //   return comp_str < 0;
      // };

      // if (a->TributaryLayer == b->TributaryLayer) {
      //   if (a->MinDistance2Transfer == b->MinDistance2Transfer) {
      //     if (a->MinLayer == b->MinLayer) {
      //       // Put the task with the same names together
      //       auto comp_str = a->node->VisualStr().compare(b->node->VisualStr());
      //       if (comp_str == 0) {
      //         // the order does not matter right now, but we need a strict order
      //         return a < b;
      //       } else {
      //         return comp_str < 0;
      //       }
      //     } else {
      //       // the node that shows up first has higher priority
      //       return a->MinLayer < b->MinLayer;
      //     }
      //   } else {
      //     return a->MinDistance2Transfer < b->MinDistance2Transfer;
      //   }
      // } else {
      //   return a->TributaryLayer < b->TributaryLayer;
      // }
    }
  };

  // Classify sets for the task nodes
  // std::set<TopoStruct*, comp> waiting_transfer; // 0
  // std::set<TopoStruct*, comp> waiting_computation; // 1
  // std::set<TopoStruct*, comp> run_asap;  // 2, run as soon as possible
  // std::set<TopoStruct*, comp> run_alap;  // 3, run as late as possible
  std::vector<std::set<TopoStruct*, comp>> waiting_lists(4);

  std::vector<int32_t> remain_task_nums(4, 0);

  // Classifier for the set according to the task type
  auto set_classifier = [&](TaskNode* node) {
    // Check task.pb.h for detail
    int32_t task_type = node->GetTaskType();
    if (task_type == 1) { return 1; }
    if (task_type == 12 || task_type == 13 || (48 <= task_type && task_type <= 64)) { return 0; }
    if (task_type == 47) { return 2; }
    return 3;
  };

  HashMap<int32_t, HashMap<int32_t, std::set<TopoStruct*, comp>>> task_type2machine_id2topo_structs;

  auto SetOrderInGraph = [&](TaskNode* task_node) {
    if (GlobalProcessCtx::Rank() == 0) {
      auto& topo_struct = task_node2topo_struct[task_node];
      std::cout << "Execution order: " << order_in_graph << ": " << task_node->VisualStr()
                << ", node id: " << task_node->node_id() << std::endl;
      std::cout << ": task type: " << task_node->GetTaskType() << ", "
                << (task_node->parallel_ctx() == 0) << ", MinLayer: " << topo_struct.MinLayer
                << ", TributaryLayer: " << topo_struct.TributaryLayer
                << ", MinDist2Transfer: " << topo_struct.MinDistance2Transfer
                << ", machine id: " << task_node->machine_id()
                << ", thread id: " << task_node->thrd_id() << std::endl;

      if (task_type_map.find(task_node->GetTaskType()) == task_type_map.end()) {
        task_type_map[task_node->GetTaskType()] = 0;
      }
      task_type_map[task_node->GetTaskType()]++;
    }
    task_node->set_order_in_graph(order_in_graph);
    ordered_task_nodes_.emplace_back(task_node);
    ++order_in_graph;
  };

  // wait in the list
  auto wait = [&](TaskNode* node) {
    TopoStruct* topo_struct = &task_node2topo_struct[node];
    waiting_lists[set_classifier(node)].insert(topo_struct);
    task_type2machine_id2topo_structs[node->GetTaskType()][node->machine_id()].insert(topo_struct);
  };

  std::map<int32_t, std::map<int32_t, int32_t>> task_type2node_id2machine_id;
  // initialization
  HashMap<TaskNode*, int32_t> counter_in;
  ForEachNode([&](TaskNode* node) {
    int32_t count = node->in_edges().size();
    counter_in[node] = count;
    if (count == 0) { wait(node); }
    remain_task_nums[set_classifier(node)]++;
    task_type2node_id2machine_id[node->GetTaskType()][node->node_id()] = node->machine_id();
  });

  for (auto& task_type_group : task_type2node_id2machine_id) {
    std::cout << "task type: " << task_type_group.first << std::endl;
    int32_t pre_machine_id = -1;
    for (auto& pair : task_type_group.second) {
      std::cout << "node id: " << pair.first << ", machine id: " << pair.second << ", ? "
                << (pair.second == 0 || pair.second > pre_machine_id) << std::endl;
      pre_machine_id = pair.second;
    }
  }

  if (GlobalProcessCtx::Rank() == 0) {
    std::cout << "Total task nums:" << std::endl;
    std::cout << "Transfers: " << remain_task_nums[0] << ", Computation: " << remain_task_nums[1]
              << ", Run Asap: " << remain_task_nums[2] << ", Run Alap: " << remain_task_nums[3]
              << std::endl;
  }

  // Finish execution
  auto finish_execution = [&](TaskNode* node) {
    node->ForEachNodeOnOutEdge([&](TaskNode* out) {
      if (--counter_in[out] == 0) { wait(out); }
    });
  };

  // The same order in the set
  auto should_run_simultaneously = [](TopoStruct* a, TopoStruct* b) -> bool {
    // Normal node would have the same name
    if (a->node->GetTaskType() == 1) { return a->node->VisualStr() == b->node->VisualStr(); }
    // Otherwise they must have the same parameters with different machine ids and the closest node
    // id
    return a->MinLayer == b->MinLayer && a->TributaryLayer == b->TributaryLayer
           && a->MinDistance2Transfer == b->MinDistance2Transfer;
  };

  // Move the first node of the waiting list to the execution list
  auto move2execution_list = [&](std::set<TopoStruct*, comp>& waiting_list,
                                 std::vector<TaskNode*>& execution_list) {
    TaskNode* first_node = (*waiting_list.begin())->node;
    int32_t execution_num = 0;
    TopoStruct* target_topo_struct = &task_node2topo_struct[first_node];
    // Find all the same nodes in different machine
    // They should be run simultaneously
    for (auto& machine_id2topo_structs :
         task_type2machine_id2topo_structs[first_node->GetTaskType()]) {
      auto& topo_structs = machine_id2topo_structs.second;
      if (topo_structs.empty()) { continue; }
      TopoStruct* first_topo_struct = *topo_structs.begin();
      if (should_run_simultaneously(target_topo_struct, first_topo_struct)) {
        execution_num++;
        execution_list.push_back(first_topo_struct->node);
        waiting_list.erase(first_topo_struct);
        // topo_structs.erase(first_topo_struct);
        topo_structs.erase(topo_structs.begin());
      }
    }
    CHECK_GT(execution_num, 0) << "Error, no task nodes are moved to the execution list";
  };

  // Execute the first n nodes in the waiting list
  auto execute = [&](int32_t list_classifier, int32_t n, bool if_reverse = false) {
    // n>=1
    if (n <= 0) { return; }
    if (GlobalProcessCtx::Rank() == 0) {
      std::cout << "Total task nums:" << std::endl;
      std::cout << "Transfers: " << waiting_lists[0].size()
                << ", Computation: " << waiting_lists[1].size()
                << ", Run Asap: " << waiting_lists[2].size()
                << ", Run Alap: " << waiting_lists[3].size() << std::endl;
    }
    auto& waiting_list = waiting_lists[list_classifier];
    std::vector<TaskNode*> execution_list;
    int32_t count = 0;
    // Move to the execution list
    while (!waiting_list.empty()) {
      move2execution_list(waiting_list, execution_list);
      count++;
      if (count >= n) { break; }
    }
    remain_task_nums[list_classifier] -= execution_list.size();
    // Set the order and then remove from the execution list
    for (auto* node : execution_list) {
      SetOrderInGraph(node);
      finish_execution(node);
    }
  };

  // int32_t max_overlap_computation_num = ParseIntegerFromEnv("MAX_OVERLAP_NUM", 40);

  // straightening
  while (true) {
    if (waiting_lists[2].empty()) {
      if (waiting_lists[0].empty()) {
        if (waiting_lists[1].empty()) {
          if (waiting_lists[3].empty()) {
            if (GlobalProcessCtx::Rank() == 0) { std::cout << "Execution done" << std::endl; }
            break;
          } else {
            execute(3, waiting_lists[3].size());
          }
        } else {
          execute(1, 1);
        }
      } else {
        int32_t computation_num =
            std::min(int32_t(waiting_lists[1].size() / (waiting_lists[0].size())),
                     remain_task_nums[1] / remain_task_nums[0]);
        // Holding the transfer
        std::vector<TaskNode*> transfer_execution_list;
        move2execution_list(waiting_lists[0], transfer_execution_list);
        remain_task_nums[0] -= transfer_execution_list.size();
        for (auto* transfer_node : transfer_execution_list) { SetOrderInGraph(transfer_node); }
        // Overlap transfer with computation
        execute(1, computation_num);

        // Release the transfer
        for (auto* transfer_node : transfer_execution_list) { finish_execution(transfer_node); }
      }
    } else {
      execute(2, waiting_lists[2].size());
    }
  }

  // test debug
  if (GlobalProcessCtx::Rank() == 0) {
    std::cout << "Print all task type: " << std::endl;
    for (auto& pair : task_type_map) {
      std::cout << "task type: " << pair.first << ", " << pair.second << std::endl;
    }
  }
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
    if (task_node->device_type() != DeviceType::kCUDA) { return; }
    int64_t dev_phy_id = task_node->stream_id().device_id().device_index();
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
    const NdSbp& src_nd_sbp = src_op_node->NdSbp4Lbi(lbi);
    const NdSbp& dst_nd_sbp = dst_op_node->NdSbp4Lbi(lbi);
    const ParallelDesc& src_parallel_desc = src_op_node->parallel_desc();
    const ParallelDesc& dst_parallel_desc = dst_op_node->parallel_desc();
    const BlobDesc& blob_desc = src_op_node->LogicalBlobDesc4Lbi(lbi);
    auto status = CHECK_JUST(hierarchical_sub_tsk_gph_builder_->Build(
        sub_tsk_gph_builder_ctx_.get(), in_nodes, &out_nodes, &sorted_ctrl_tasks, src_parallel_desc,
        dst_parallel_desc, lbi, blob_desc, src_nd_sbp, dst_nd_sbp,
        *(CHECK_JUST(src_op_node->op().GetOpTimeShape()).get())));
    boxing_logger_->Log(*status, src_op_node->op().op_name(), dst_op_node->op().op_name(),
                        src_parallel_desc, dst_parallel_desc, src_nd_sbp, dst_nd_sbp, lbi,
                        blob_desc);
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
    for (const LogicalBlobId& lbi : op_edge->lbis()) { ConnectWithLbi(src, dst, lbi); }
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
  TaskNode* proxy_node = GetProxyNode(src_node, lbi, dst_node->MemZoneId121());
  ConnectWithLbi(proxy_node, dst_node, lbi);
}

}  // namespace oneflow
