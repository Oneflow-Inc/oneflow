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
#include "oneflow/core/common/env_var/debug_mode.h"
#include "oneflow/core/graph/inplace_lbi_graph.h"
#include "oneflow/core/job/job_conf.pb.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/register/blob_desc.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/operator/variable_op.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/graph/normal_forward_compute_task_node.h"
#include "oneflow/core/graph/boxing_identity_task_node.h"
#include "oneflow/core/job/scope.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"
#include "oneflow/core/vm/symbol_storage.h"
#include "oneflow/core/job_rewriter/calculation_pass.h"
#include "oneflow/core/graph/boxing/sub_task_graph_builder_util.h"
#include "oneflow/core/graph/boxing/hierarchical_sub_task_graph_builder_impl.h"
#include "oneflow/core/graph/task_stream_index_manager.h"
#include "oneflow/core/ep/include/primitive/memcpy.h"
#include "oneflow/core/graph/straighten_nodes.h"
#include "oneflow/core/register/runtime_register_desc.h"
#include "oneflow/core/common/env_var/env_var.h"
#include "oneflow/core/framework/user_op_registry_manager.h"

namespace oneflow {

// TODO(Chengcheng): default false.
DEFINE_ENV_BOOL(ONEFLOW_ENABLE_OUTDATED_OPT_FW_CHAIN_MERGE, true);

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

bool IsSubsetTickOpConf(const OperatorConf& op_conf) {
  return op_conf.has_src_subset_tick_conf() || op_conf.has_dst_subset_tick_conf();
}

bool IsTickOpConf(const OperatorConf& conf) {
  return IsClassRegistered<int32_t, IsTickTockOpTypeCase>(conf.op_type_case());
}

std::string GetOpConfCalculationPassName(const OperatorConf& op_conf) {
  CHECK(op_conf.has_scope_symbol_id());
  int64_t scope_symbol_id = op_conf.scope_symbol_id();
  CHECK(Singleton<symbol::Storage<Scope>>::Get()->Has(scope_symbol_id))
      << " Error! op : \n " << op_conf.DebugString()
      << " has error scope_symbol_id = " << scope_symbol_id
      << " which cannot find in Singleton<symbol::Storage<Scope>>::Get()\n";
  const Scope& scope = Singleton<symbol::Storage<Scope>>::Get()->Get(scope_symbol_id);
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
  if (!Singleton<ResourceDesc, ForSession>::Get()->nccl_use_compute_stream()
      && IsOptimizerPassOp(op) && EnvBool<ONEFLOW_ENABLE_OUTDATED_OPT_FW_CHAIN_MERGE>()) {
    return true;
  }
  return false;
}

bool IsTaskNodeProducedRegstHasMultiRegstNum(const TaskNode* node) {
  for (const auto& pair : node->produced_regsts()) {
    if (pair.second->min_register_num() > 1) { return true; }
  }
  return false;
}

bool CanBeMergedInChain(const TaskNode* node) {
  // ONLY the node which is NormalForward and in GPU and NOT variable can be merged.
  if (IsTaskNodeProducedRegstHasMultiRegstNum(node)) { return false; }
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
  CHECK(IsValidChainId(this_chain_id));
  CHECK(!IsValidChainId(this_node->chain_id()));
  // bfs search all node can be merged in this chain
  std::shared_ptr<const Shape> seed_time_shape = GetTaskNodeTimeShape(this_node);
  HashSet<TaskNode*> visited_nodes;
  std::queue<TaskNode*> queued_nodes;
  queued_nodes.push(this_node);
  visited_nodes.insert(this_node);
  while (!queued_nodes.empty()) {
    TaskNode* cur_node = queued_nodes.front();
    queued_nodes.pop();

    CHECK(!IsValidChainId(cur_node->chain_id()));
    cur_node->set_chain_id(this_chain_id);

    cur_node->ForEachNodeOnInOutDataEdge([&](TaskNode* next_node) {
      if (visited_nodes.find(next_node) == visited_nodes.end() && CanBeMergedInChain(next_node)
          && this_node->thrd_id() == next_node->thrd_id()
          && (*GetTaskNodeTimeShape(next_node)) == (*seed_time_shape)) {
        if (!IsValidChainId(next_node->chain_id())) {
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
    if (IsValidChainId(src_node->chain_id()) && IsValidChainId(dst_node->chain_id())
        && src_node->chain_id() == dst_node->chain_id()
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
  if (Singleton<ResourceDesc, ForSession>::Get()->enable_debug_mode()) {
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
        stream_index = Singleton<TaskStreamIndexManager>::Get()->GetNamedTaskStreamIndex(
            device_id, stream_name_hint);
      } else {
        stream_index = Singleton<TaskStreamIndexManager>::Get()->GetTaskStreamIndex(
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
  //   Multi-op corresponding to SAME TaskType such as:
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
        if (src_time_shape->elem_cnt() != dst_time_shape->elem_cnt()) {
          // NOTE(chengcheng): acc / pack op node can be merged and add ctrl edge.
          CHECK(src->op().op_conf().has_user_conf());
          const std::string& op_type_name = src->op().op_conf().user_conf().op_type_name();
          CHECK(op_type_name == "acc" || op_type_name == "pack");
          const Shape* src_input_time_shape =
              CHECK_JUST(src->op().GetInputBlobFastestTimeShape()).get();
          CHECK_EQ(src_input_time_shape->elem_cnt(), dst_time_shape->elem_cnt());
        } else {
          CHECK_EQ(src_time_shape->elem_cnt(), dst_time_shape->elem_cnt());
        }
        if (!src->parallel_desc().EqualsIgnoringHierarchy(dst->parallel_desc())) {
          LOG(WARNING) << " Warning, there is a ctrl edge connected across placement from: "
                       << src->op().op_name() << " ["
                       << src->parallel_desc().parallel_conf().DebugString()
                       << "] to: " << dst->op().op_name() << " ["
                       << dst->parallel_desc().parallel_conf().DebugString() << "]";
        }
        Handler(src, dst);
      }
    }
  });
}

void GetHostInputLbis4OpNode(const OpNode* op_node,
                             std::vector<LogicalBlobId>* host_mem_input_lbis) {
  host_mem_input_lbis->clear();
  if (op_node->op().op_conf().has_user_conf()) {
    const auto& user_conf = op_node->op().op_conf().user_conf();
    const auto& op_type_name = user_conf.op_type_name();
    if (user_op::UserOpHostMemoryInputRegistry::Get().HasHostMemoryInput(op_type_name)) {
      const auto& inputs = [&]() -> std::vector<std::pair<std::string, int32_t>> {
        const auto& arg_map = op_node->op().op_conf().user_conf().input();
        std::vector<std::pair<std::string, int32_t>> arg_vec;
        for (auto it = arg_map.begin(); it != arg_map.end(); ++it) {
          for (int32_t i = 0; i < it->second.s_size(); ++i) {
            arg_vec.emplace_back(std::make_pair(it->first, i));
          }
        }
        return arg_vec;
      }();
      for (const auto& pair : inputs) {
        if (user_op::UserOpHostMemoryInputRegistry::Get().IsHostMemoryInput4Op(
                op_type_name, pair.first, pair.second)) {
          const LogicalBlobId& host_input_lbi =
              GenLogicalBlobId(user_conf.input().at(pair.first).s(pair.second));
          host_mem_input_lbis->emplace_back(host_input_lbi);
        }
      }
    }
  }
}

}  // namespace

TaskGraph::TaskGraph() {
  OpGraph* op_graph = Singleton<OpGraph>::Get();
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

  if (Singleton<ResourceDesc, ForSession>::Get()->enable_debug_mode()) { ToDotWithAutoFilePath(); }
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
        copy_task->Init(CopyHdType::D2H, src_mem_zone_id, lbi);
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
      copy_task->Init(CopyHdType::H2D, dst_mem_zone_id, lbi);
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
  if (EnableLogicalChain()) {
    MergeChainByLogicalChainId();
  } else {
    // TODO(chengcheng): erase old chain version in the future.
    MergeChainByPhysicalTaskGraph();
  }
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

void TaskGraph::MergeChainByPhysicalTaskGraph() {
  int64_t chain_id = 0;
  for (auto* this_node : ordered_task_nodes_) {
    // skip if this node has been set in a chain.
    if (IsValidChainId(this_node->chain_id())) { continue; }

    if (CanBeMergedInChain(this_node)) {
      TraverseConnectedSubGraphMergeInThisChain(this_node, chain_id);
    } else {
      this_node->set_chain_id(chain_id);
    }

    ++chain_id;
  }
  for (auto* node : ordered_task_nodes_) { CHECK(IsValidChainId(node->chain_id())); }
}

void TaskGraph::MergeChainByLogicalChainId() {
  for (TaskNode* this_node : ordered_task_nodes_) {
    CompTaskNode* comp_node = dynamic_cast<CompTaskNode*>(this_node);
    if (!comp_node) { continue; }
    const int64_t logical_chain_id = comp_node->op()->op_conf().logical_chain_id();
    if (IsValidChainId(logical_chain_id)) { this_node->set_chain_id(logical_chain_id); }
  }
}

void TaskGraph::BuildCtrlRegstDescInSameChain() {
  auto GenPhysicalChainId = [](TaskNode* node) {
    // NOTE(chengcheng): different rank cannot use same chain id for bad ctrl link.
    return (node->chain_id() << 31) | (node->machine_id());
  };
  HashMap<int64_t, TaskNode*> physical_chain_id2node;
  for (auto* node : ordered_task_nodes_) {
    if (IsConnectToTickOp(node)) { continue; }
    // NOTE(chengcheng): skip invalid chain id
    if (!IsValidChainId(node->chain_id())) { continue; }
    int64_t physical_chain_id = GenPhysicalChainId(node);
    auto iter = physical_chain_id2node.find(physical_chain_id);
    if (iter == physical_chain_id2node.end()) {
      CHECK(physical_chain_id2node.emplace(physical_chain_id, node).second);
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
  if (Singleton<ResourceDesc, ForSession>::Get()->enable_debug_mode()) {
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
  std::vector<LogicalBlobId> host_mem_input_lbis;
  GetHostInputLbis4OpNode(dst_op_node, &host_mem_input_lbis);
  for (const LogicalBlobId& lbi : op_edge->lbis()) {
    std::vector<TaskNode*> in_nodes(sorted_src_comp_tasks.begin(), sorted_src_comp_tasks.end());
    std::vector<TaskNode*> out_nodes;
    out_nodes.reserve(sorted_dst_comp_tasks.size());
    std::vector<std::vector<TaskNode*>> sorted_ctrl_tasks;
    const NdSbp& src_nd_sbp = src_op_node->NdSbp4Lbi(lbi);
    const NdSbp& dst_nd_sbp = dst_op_node->NdSbp4Lbi(lbi);
    const ParallelDesc& src_parallel_desc = src_op_node->parallel_desc();
    const ParallelDesc& dst_parallel_desc = [&]() {
      if (std::find(host_mem_input_lbis.begin(), host_mem_input_lbis.end(), lbi)
          != host_mem_input_lbis.end()) {
        return *CHECK_JUST(
            ReplaceDeviceType(SymbolOf(dst_op_node->parallel_desc()), DeviceType::kCPU));
      } else {
        return dst_op_node->parallel_desc();
      }
    }();
    const BlobDesc& blob_desc = src_op_node->LogicalBlobDesc4Lbi(lbi);
    VLOG(3) << "src op: " << src_op_node->op().op_name()
            << " dst op: " << dst_op_node->op().op_name()
            << " src_parallel_conf: " << src_parallel_desc.parallel_conf().DebugString()
            << " dst parallel conf: " << dst_parallel_desc.parallel_conf().DebugString()
            << " src_nd_sbp " << src_nd_sbp.DebugString() << " dst nd_sbp "
            << dst_nd_sbp.DebugString();
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
  std::vector<LogicalBlobId> host_mem_input_lbis;
  GetHostInputLbis4OpNode(op_edge->dst_node(), &host_mem_input_lbis);
  CHECK_EQ(sorted_src_comp_tasks.size(), sorted_dst_comp_tasks.size());
  FOR_RANGE(size_t, i, 0, sorted_src_comp_tasks.size()) {
    for (const LogicalBlobId& lbi : op_edge->lbis()) {
      bool is_host_mem_input =
          std::find(host_mem_input_lbis.begin(), host_mem_input_lbis.end(), lbi)
          != host_mem_input_lbis.end();
      BuildTaskPath(sorted_src_comp_tasks.at(i), sorted_dst_comp_tasks.at(i), lbi,
                    is_host_mem_input);
    }
  }
}

DEFINE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphByBroadcastToBroadcast) {
  std::vector<LogicalBlobId> host_mem_input_lbis;
  GetHostInputLbis4OpNode(op_edge->dst_node(), &host_mem_input_lbis);
  for (CompTaskNode* dst_node : sorted_dst_comp_tasks) {
    CompTaskNode* nearest_src_node =
        SubTskGphBuilderUtil::FindNearestNode(sorted_src_comp_tasks, dst_node);
    CHECK_NOTNULL(nearest_src_node);
    for (const LogicalBlobId& lbi : op_edge->lbis()) {
      bool is_host_mem_input =
          std::find(host_mem_input_lbis.begin(), host_mem_input_lbis.end(), lbi)
          != host_mem_input_lbis.end();
      BuildTaskPath(nearest_src_node, dst_node, lbi, is_host_mem_input);
    }
  }
}

DEFINE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphByPartialInLbiConnect) {
  const Operator& src_op = op_edge->src_node()->op();
  const Operator& dst_op = op_edge->dst_node()->op();
  HashSet<LogicalBlobId> lbis;
  std::vector<LogicalBlobId> host_mem_input_lbis;
  GetHostInputLbis4OpNode(op_edge->dst_node(), &host_mem_input_lbis);
  for (const auto& obn : src_op.output_bns()) { lbis.insert(src_op.BnInOp2Lbi(obn)); }
  CHECK_EQ(sorted_src_comp_tasks.size(), 1);
  CHECK_EQ(dst_op.input_bns().size(), sorted_dst_comp_tasks.size());
  FOR_RANGE(int, i, 0, sorted_dst_comp_tasks.size()) {
    const auto& lbi = dst_op.BnInOp2Lbi(dst_op.input_bns().Get(i));
    if (lbis.find(lbi) != lbis.end()) {
      bool is_host_mem_input =
          std::find(host_mem_input_lbis.begin(), host_mem_input_lbis.end(), lbi)
          != host_mem_input_lbis.end();
      BuildTaskPath(sorted_src_comp_tasks.at(0), sorted_dst_comp_tasks.at(i), lbi,
                    is_host_mem_input);
    }
  }
}

DEFINE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphByPartialOutLbiConnect) {
  const Operator& src_op = op_edge->src_node()->op();
  const Operator& dst_op = op_edge->dst_node()->op();
  HashSet<LogicalBlobId> lbis;
  std::vector<LogicalBlobId> host_mem_input_lbis;
  GetHostInputLbis4OpNode(op_edge->dst_node(), &host_mem_input_lbis);
  for (const auto& ibn : dst_op.input_bns()) { lbis.insert(dst_op.BnInOp2Lbi(ibn)); }
  CHECK_EQ(sorted_dst_comp_tasks.size(), 1);
  CHECK_EQ(src_op.output_bns().size(), sorted_src_comp_tasks.size());
  FOR_RANGE(int, i, 0, sorted_src_comp_tasks.size()) {
    const auto& lbi = src_op.BnInOp2Lbi(src_op.output_bns().Get(i));
    if (lbis.find(lbi) != lbis.end()) {
      bool is_host_mem_input =
          std::find(host_mem_input_lbis.begin(), host_mem_input_lbis.end(), lbi)
          != host_mem_input_lbis.end();
      BuildTaskPath(sorted_src_comp_tasks.at(i), sorted_dst_comp_tasks.at(0), lbi,
                    is_host_mem_input);
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

void TaskGraph::BuildTaskPath(TaskNode* src_node, TaskNode* dst_node, const LogicalBlobId& lbi,
                              bool is_host_mem_input) {
  const MemZoneId dst_mem_zone_id = [&]() {
    if (is_host_mem_input) {
      MemZoneId mem_zone_id = dst_node->MemZoneId121();
      return MemZoneId(mem_zone_id.rank(), DeviceType::kCPU, 0);
    } else {
      return dst_node->MemZoneId121();
    }
  }();
  TaskNode* proxy_node = GetProxyNode(src_node, lbi, dst_mem_zone_id);
  ConnectWithLbi(proxy_node, dst_node, lbi);
}

void TaskGraph::DecideExecutionOrder() {
  // For one machine with no transfer available, the straighten algorithm for overlaps consume a lot
  // of memory
  StraightenAlgorithmTag straighten_algorithm_tag =
      GlobalJobDesc().job_conf().straighten_algorithm_tag_in_task_graph();
  if (straighten_algorithm_tag == StraightenAlgorithmTag::kDisableStraighten
      || (straighten_algorithm_tag == StraightenAlgorithmTag::kOverlap4Transfer
          && GlobalProcessCtx::WorldSize() == 1)) {
    SetOrderInGraphForEachNode();
  } else {
    StraightenNodes(this, &ordered_task_nodes_,
                    Singleton<ResourceDesc, ForSession>::Get()->nccl_use_compute_stream());
  }
}

}  // namespace oneflow
