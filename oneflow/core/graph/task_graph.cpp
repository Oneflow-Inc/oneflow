#include "oneflow/core/graph/task_graph.h"
#include "oneflow/core/graph/chain_graph.h"
#include "oneflow/core/graph/boxing_task_node.h"
#include "oneflow/core/graph/copy_task_node.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

TaskGraph::TaskGraph(std::unique_ptr<const LogicalGraph>&& logical_gph) {
  logical_gph_ = std::move(logical_gph);
  HashMap<const LogicalNode*, std::vector<CompTaskNode*>> logical2sorted_comp_tasks;
  HashMap<const LogicalNode*, std::vector<TaskNode*>> logical2sorted_in_box;
  HashMap<const LogicalNode*, std::vector<TaskNode*>> logical2sorted_out_box;
  HashMap<CompTaskNode*, HashMap<int64_t, std::vector<TaskNode*>>> buf121;
  const JobDesc* job_desc = Global<JobDesc>::Get();
  auto Mut121BufTask = [&](CompTaskNode* task_node, int64_t machine_id, int32_t mem_zone_id) {
    auto& buf_vec = buf121[task_node][machine_id];
    if (buf_vec.empty()) { buf_vec.assign(job_desc->MemZoneNum(), nullptr); }
    return &(buf_vec.at(mem_zone_id));
  };

  std::vector<int64_t> cpu_device_offset(job_desc->TotalMachineNum(), 0);
  std::vector<int64_t> persistence_offset(job_desc->TotalMachineNum(), 0);
  auto AllocateCpuThrdIdEvenly = [&](const TaskNode* task_node) {
    int64_t ret = -1;
    if (task_node->IsPersistence() == false) {
      int64_t& offset = cpu_device_offset.at(task_node->machine_id());
      ret = Global<IDMgr>::Get()->GetCpuDeviceThrdId(offset);
      offset = (offset + 1) % job_desc->CpuDeviceNum();
    } else {
      int64_t& offset = persistence_offset.at(task_node->machine_id());
      ret = Global<IDMgr>::Get()->GetPersistenceThrdId(offset);
      offset = (offset + 1) % job_desc->PersistenceWorkerNum();
    }
    return ret;
  };
  logical_gph_->ForEachNode([&](const LogicalNode* logical_node) {
    logical_node->GenSortedCompTaskNodes(
        AllocateCpuThrdIdEvenly, [&](CompTaskNode* comp_task_node) {
          AddAllocatedNode(comp_task_node);
          logical2sorted_comp_tasks[logical_node].push_back(comp_task_node);
          comp_task_node->set_area_id(logical_node->GetAreaId());
        });
  });
  logical_gph_->ForEachEdge([&](const LogicalEdge* logical_edge) {
    BldSubTskGphMthd method =
        GetMthdForBldSubTskGph(logical_edge->src_node(), logical_edge->dst_node());
    (this->*method)(logical_edge->src_node(), logical_edge->dst_node(),
                    logical2sorted_comp_tasks.at(logical_edge->src_node()),
                    logical2sorted_comp_tasks.at(logical_edge->dst_node()), &logical2sorted_in_box,
                    &logical2sorted_out_box, Mut121BufTask, AllocateCpuThrdIdEvenly);
    SetAreaIdForNewNodes(logical_edge->src_node(), logical_edge->dst_node());
  });
  ToDotWithAutoFilePath();
}

void TaskGraph::FindChainsInSameStream() {
  CollectAncestorsForEachNode();

  ChainGraph chain_gph(*this);
  const auto& ordered_chain_nodes = chain_gph.ordered_chain_nodes();
  int64_t order_in_graph = 0;
  HashMap<int64_t, HashSet<TaskNode*>> chain_id2task_nodes;
  for (auto& chain_node : ordered_chain_nodes) {
    auto& ordered_in_chain = chain_node->chain_it()->nodes;
    int64_t chain_id = chain_node->chain_id();
    LOG(INFO) << "chain_id" << chain_id;
    for (auto& task_node : ordered_in_chain) {
      task_node->set_chain_id(chain_id);
      task_node->set_order_in_graph(order_in_graph);
      CHECK(chain_id2task_nodes[chain_id].insert(task_node).second);
      ordered_task_nodes_.emplace_back(task_node);
      ++order_in_graph;
      LOG(INFO) << task_node->VisualStrWithOpName();
    }
  }
}

void TaskGraph::AddOrderingCtrlEdgeInSameChain() {
  FindChainsInSameStream();

  HashMap<int64_t, TaskNode*> chain_id2node;
  for (auto node : ordered_task_nodes_) {
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

void TaskGraph::AddMutexCtrlEdgeInSameChain() { UNIMPLEMENTED(); }

void TaskGraph::AddOrderCtrlEdgeBetweenCopyAndMdUpdt() { UNIMPLEMENTED(); }

void TaskGraph::CollectAncestorsForEachNode() {
  std::vector<TaskNode*> ordered_nodes;
  AcyclicTopoForEachNode([&](TaskNode* node) { ordered_nodes.emplace_back(node); });
  for (auto it = ordered_nodes.begin(); it != ordered_nodes.end(); ++it) {
    TaskNode* task_node = *it;
    task_node->mut_ancestors().clear();
    task_node->ForEachNodeOnInEdge([&](TaskNode* node_on_in_edge) {
      if (IsBackEdge(node_on_in_edge, task_node)) return;
      task_node->mut_ancestors().insert(node_on_in_edge->ancestors().begin(),
                                        node_on_in_edge->ancestors().end());
      task_node->mut_ancestors().insert(node_on_in_edge);

    });
  }
}

void TaskGraph::AcyclicTopoForEachNode(std::function<void(TaskNode* node)> handler) const {
  std::list<TaskNode*> starts;
  ForEachNode([&](TaskNode* node) {
    if (node->consumed_regsts().empty() && !node->IsMeaningLess()) { starts.push_back(node); }
  });
  auto ForEachInNode = [&](TaskNode* node, const std::function<void(TaskNode*)>& handler) {
    node->ForEachNodeOnInEdge([&](TaskNode* node_on_in_edge) {
      if (IsBackEdge(node_on_in_edge, node)) return;
      handler(const_cast<TaskNode*>(node_on_in_edge));
    });
  };
  auto ForEachOutNode = [&](TaskNode* node, const std::function<void(TaskNode*)>& handler) {
    std::vector<TaskNode*> out_nodes;
    node->ForEachNodeOnOutEdge([&](TaskNode* node_on_out_edge) {
      if (IsBackEdge(node, node_on_out_edge)) return;
      // handler(const_cast<TaskNode*>(node_on_out_edge));
      out_nodes.emplace_back(node_on_out_edge);
    });
    // std::sort(out_nodes.begin(), out_nodes.end(), [](TaskNode* lhs, TaskNode* rhs) {
    //   return (rhs->GetTaskType() == kLossAcc || rhs->GetTaskType() == kMdDiffAcc);
    //});
    LOG(INFO) << "OUT:" << node->VisualStrWithOpName();
    for (auto& out_node : out_nodes) {
      handler(const_cast<TaskNode*>(out_node));
      LOG(INFO) << out_node->VisualStrWithOpName();
    }
  };
  DfsTopoForEachNode(starts, ForEachInNode, ForEachOutNode, handler);
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
                   AllocateCpuThrdIdEvenly);
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
    Build121Path(src, dst, Mut121BufTask, true);
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
                         nullptr, Mut121BufTask, AllocateCpuThrdIdEvenly);
}

DEFINE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphByReduceScatter2ReduceAdd) {
  for (CompTaskNode* src_comp_task : sorted_src_comp_tasks) {
    for (CompTaskNode* dst_comp_task : sorted_dst_comp_tasks) {
      ConnectWithCopyCommNetIfNeed(src_comp_task, dst_comp_task);
    }
  }
}

DEFINE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphByReduceAdd2ReduceGather) {
  CHECK_GE(sorted_src_comp_tasks.size(), 2);
  for (CompTaskNode* src_comp_task : sorted_src_comp_tasks) {
    TaskNode* src_d2h_task = src_comp_task;
    if (src_comp_task->device_type() == DeviceType::kGPU) {
      src_d2h_task = AddCopyD2HTaskFrom(src_comp_task);
      Connect<TaskNode>(src_comp_task, NewEdge(), src_d2h_task);
    }
    for (CompTaskNode* dst_comp_task : sorted_dst_comp_tasks) {
      if (src_comp_task->parallel_id() == dst_comp_task->parallel_id()) {
        Connect<TaskNode>(src_comp_task, NewEdge(), dst_comp_task);
      } else {
        ConnectWithCopyCommNetIfNeed(src_d2h_task, dst_comp_task);
      }
    }
  }
}

void TaskGraph::Build121Path(
    CompTaskNode* src, CompTaskNode* dst,
    std::function<TaskNode**(CompTaskNode* src, int64_t machine_id, int32_t mem_zone_id)>
        Mut121BufTask,
    bool allow_share_path) {
  CHECK_NE(src, dst);
  auto Get121BufTask = [&](int64_t machine_id, int32_t mem_zone_id) {
    return *Mut121BufTask(src, machine_id, mem_zone_id);
  };
  auto Set121BufTask = [&](int64_t machine_id, int32_t mem_zone_id, TaskNode* new_val) {
    TaskNode** cur_val = Mut121BufTask(src, machine_id, mem_zone_id);
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
    cur_node = Build121Step(cur_node, dst, Get121BufTask, Set121BufTask, allow_share_path);
  }
  Connect<TaskNode>(cur_node, NewEdge(), dst);
}

TaskNode* TaskGraph::Build121Step(
    TaskNode* cur_node, TaskNode* dst,
    std::function<TaskNode*(int64_t machine_id, int32_t mem_zone_id)> Get121BufTask,
    std::function<TaskNode*(int64_t machine_id, int32_t mem_zone_id, TaskNode*)> Set121BufTask,
    bool allow_share_path) {
  int32_t cpu_mem_zone_id = Global<IDMgr>::Get()->CpuMemZoneId();
  int32_t next_mem_zone_id = -1;
  TaskNode* next_node = nullptr;
  if (cur_node->MemZoneId121() != cpu_mem_zone_id) {
    next_mem_zone_id = cpu_mem_zone_id;
    if (!allow_share_path
        || !(next_node = Get121BufTask(cur_node->machine_id(), next_mem_zone_id))) {
      next_node = AddCopyD2HTaskFrom(cur_node);
      Connect<TaskNode>(cur_node, NewEdge(), next_node);
    }
  } else if (cur_node->machine_id() == dst->machine_id()) {
    next_mem_zone_id = dst->MemZoneId121();
    if (!allow_share_path
        || !(next_node = Get121BufTask(cur_node->machine_id(), next_mem_zone_id))) {
      next_node = AddCopyH2DTaskTo(dst);
      Connect<TaskNode>(cur_node, NewEdge(), next_node);
    }
  } else if (cur_node->machine_id() != dst->machine_id()) {
    next_mem_zone_id = cpu_mem_zone_id;
    if (!allow_share_path || !(next_node = Get121BufTask(dst->machine_id(), next_mem_zone_id))) {
      next_node = AddCopyCommNetTaskBetween(cur_node, dst);
      Connect<TaskNode>(cur_node, NewEdge(), next_node);
    }
  } else {
    UNIMPLEMENTED();
  }
  if (allow_share_path) { Set121BufTask(next_node->machine_id(), next_mem_zone_id, next_node); }
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

void TaskGraph::BuildOutBoxing(const LogicalNode* logical,
                               const std::vector<CompTaskNode*>& sorted_comp_tasks,
                               std::vector<TaskNode*>* sorted_out_box,
                               std::function<int64_t(const TaskNode*)> AllocateCpuThrdIdEvenly) {
  std::map<int64_t, std::vector<TaskNode*>> machine_id2bound_task;
  for (CompTaskNode* comp_task : sorted_comp_tasks) {
    TaskNode* task = comp_task;
    if (task->device_type() == DeviceType::kGPU) {
      task = AddCopyD2HTaskFrom(comp_task);
      Connect<TaskNode>(comp_task, NewEdge(), task);
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
