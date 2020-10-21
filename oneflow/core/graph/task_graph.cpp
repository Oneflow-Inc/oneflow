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
#include "oneflow/core/graph/chain_graph.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/graph/inplace_lbi_graph.h"
#include "oneflow/core/register/runtime_blob_desc.h"
#include "oneflow/core/job/thrd_id_generator.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/operator/variable_op.h"
#include "oneflow/core/operator/user_op_util.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/graph/boxing/sub_task_graph_builder_context.h"
#include "oneflow/core/graph/boxing/sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/chain_sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/collective_boxing_sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/slice_boxing_sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/naive_b2b_sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/b21_sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/one_to_one_sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/to_interface_sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/sub_task_graph_builder_util.h"
#include "oneflow/core/graph/boxing_identity_compute_task_node.h"

namespace oneflow {

namespace {

bool IsInterfaceTask(const TaskNode* node) {
  const auto* comp_task_node = dynamic_cast<const CompTaskNode*>(node);
  if (comp_task_node == nullptr) { return false; }
  if (comp_task_node->logical_node()->op_vec().size() != 1) { return false; }
  auto op_type_case = comp_task_node->logical_node()->SoleOp()->op_conf().op_type_case();
  return IsClassRegistered<int32_t, IsInterfaceOpConf4OpTypeCase>(op_type_case);
}

bool IsConnectToTickOp(const TaskNode* node) {
  const auto* comp_task_node = dynamic_cast<const CompTaskNode*>(node);
  if (comp_task_node == nullptr) { return false; }
  if (comp_task_node->logical_node()->op_vec().size() != 1) { return false; }
  const Operator* op = comp_task_node->logical_node()->SoleOp().get();
  if (dynamic_cast<const VariableOp*>(op) != nullptr) { return true; }
  return false;
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

}  // namespace

TaskGraph::TaskGraph(std::unique_ptr<const LogicalGraph>&& logical_gph) {
  logical_gph_ = std::move(logical_gph);
  sub_tsk_gph_builder_ctx_.reset(new SubTskGphBuilderCtx(this));
  boxing_logger_ = CreateBoxingLogger();
  std::vector<std::shared_ptr<SubTskGphBuilder>> builders;
  builders.emplace_back(new ToInterfaceSubTskGphBuilder());
  builders.emplace_back(new OneToOneSubTskGphBuilder());
  builders.emplace_back(new B21SubTskGphBuilder());
  builders.emplace_back(new CollectiveBoxingSubTskGphBuilder());
  builders.emplace_back(new SliceBoxingSubTskGphBuilder());
  builders.emplace_back(new NaiveB2BSubTskGphBuilder());
  sub_tsk_gph_builder_.reset(new ChainSubTskGphBuilder(builders));
  HashMap<const LogicalNode*, std::vector<CompTaskNode*>> logical2sorted_comp_tasks;
  HashMap<const LogicalNode*, std::vector<TaskNode*>> logical2sorted_in_box;
  HashMap<const LogicalNode*, std::vector<TaskNode*>> logical2sorted_out_box;
  HashMap<CompTaskNode*, HashMap<int64_t, std::vector<TaskNode*>>> buf_task;
  auto MutBufTask = [&](CompTaskNode* task_node, int64_t machine_id, int32_t mem_zone_id) {
    auto& buf_vec = buf_task[task_node][machine_id];
    if (buf_vec.empty()) {
      buf_vec.assign(Global<ResourceDesc, ForSession>::Get()->MemZoneNum(), nullptr);
    }
    return &(buf_vec.at(mem_zone_id));
  };

  std::vector<int64_t> cpu_device_offset(Global<ResourceDesc, ForSession>::Get()->TotalMachineNum(),
                                         0);
  auto AllocateCpuThrdIdEvenly = [&](const TaskNode* task_node) {
    CHECK(!task_node->IsIndependent());
    int64_t& offset = cpu_device_offset.at(task_node->machine_id());
    int64_t ret = Global<IDMgr>::Get()->GetCpuDeviceThrdId(offset);
    offset = (offset + 1) % Global<ResourceDesc, ForSession>::Get()->CpuDeviceNum();
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
  if (Global<ResourceDesc, ForSession>::Get()->enable_debug_mode()) { ToDotWithAutoFilePath(); }
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
  for (const auto& pair : persistence_nodes) {
    int64_t thrd_id = generator.GenerateThrdId(pair.first, pair.second->GetTaskType());
    pair.second->set_thrd_id(thrd_id);
  }
}

void TaskGraph::AcyclicTopoForEachNode(std::function<bool(TaskNode* node)> IsAllowedStartNode,
                                       const std::function<void(TaskNode* node)>& Handler) const {
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

void TaskGraph::AcyclicTopoForEachNode(const std::function<void(TaskNode* node)>& Handler) const {
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
  int64_t order_in_graph = 0;
  AcyclicTopoForEachNode([&](TaskNode* task_node) {
    task_node->set_chain_id(chain_graph.ChainId4TaskNode(task_node));
    task_node->set_order_in_graph(order_in_graph);
    ordered_task_nodes_.emplace_back(task_node);
    ++order_in_graph;
  });
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
      const UserOpCtx* user_op_ctx = dynamic_cast<const UserOpCtx*>(op_ctx);
      CHECK_NOTNULL(user_op_ctx);
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
  const std::vector<LogicalBlobId> lbis = src_logical->GetLbisTo(dst_logical);
  for (const LogicalBlobId& lbi : lbis) {
    std::vector<CompTaskNode*> src_nodes;
    if (lbis.size() == 1) {
      src_nodes = sorted_src_comp_tasks;
    } else {
      for (CompTaskNode* src_node : sorted_src_comp_tasks) {
        auto* identity_node = NewNode<BoxingIdentityCompTaskNode>();
        identity_node->Init(src_node, lbi);
        Connect<TaskNode>(src_node, NewEdge(), identity_node);
        src_nodes.push_back(identity_node);
      }
    }
    const SbpParallel& src_sbp_parallel =
        Global<OpGraph>::Get()->GetSbpParallel(src_logical->SoleOp()->op_name(), lbi);
    const SbpParallel& dst_sbp_parallel =
        Global<OpGraph>::Get()->GetSbpParallel(dst_logical->SoleOp()->op_name(), lbi);
    const std::shared_ptr<const ParallelDesc>& src_parallel_desc = src_logical->parallel_desc();
    const std::shared_ptr<const ParallelDesc>& dst_parallel_desc = dst_logical->parallel_desc();
    const BlobDesc& blob_desc = Global<OpGraph>::Get()->GetLogicalBlobDesc(lbi);
    auto status = CHECK_JUST(sub_tsk_gph_builder_->Build(
        sub_tsk_gph_builder_ctx_.get(), src_nodes, sorted_dst_comp_tasks, *src_parallel_desc,
        *dst_parallel_desc, lbi, blob_desc, src_sbp_parallel, dst_sbp_parallel));
    boxing_logger_->Log(*status);
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
  for (CompTaskNode* dst_node : sorted_dst_comp_tasks) {
    CompTaskNode* nearest_src_node =
        SubTskGphBuilderUtil::FindNearestNode(sorted_src_comp_tasks, dst_node);
    CHECK_NOTNULL(nearest_src_node);
    BuildTaskPath(nearest_src_node, dst_node, MutBufTask, true);
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

DEFINE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphNormalForwardToDecodeH2D) {
  CHECK_EQ(sorted_src_comp_tasks.size(), sorted_dst_comp_tasks.size());
  FOR_RANGE(size_t, i, 0, sorted_src_comp_tasks.size()) {
    CompTaskNode* src = sorted_src_comp_tasks.at(i);
    CompTaskNode* dst = sorted_dst_comp_tasks.at(i);
    Connect<TaskNode>(src, NewEdge(), dst);
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
    const std::function<TaskNode*(int64_t machine_id, int32_t mem_zone_id)>& GetBufTask,
    const std::function<TaskNode*(int64_t machine_id, int32_t mem_zone_id, TaskNode*)>& SetBufTask,
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
  if (IsClassRegistered<int32_t, TickTockTaskType>(task->GetTaskType())) { return nullptr; }
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
