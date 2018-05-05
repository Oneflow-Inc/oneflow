#include "oneflow/core/graph/task_graph.h"
#include "oneflow/core/graph/boxing_task_node.h"
#include "oneflow/core/graph/copy_task_node.h"

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
  auto AllocateCpuThrdId = [&](const TaskNode* task_node) {
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
    logical_node->GenSortedCompTaskNodes(AllocateCpuThrdId, [&](CompTaskNode* comp_task_node) {
      AddAllocatedNode(comp_task_node);
      logical2sorted_comp_tasks[logical_node].push_back(comp_task_node);
    });
  });
  logical_gph_->ForEachEdge([&](const LogicalEdge* logical_edge) {
    BldSubTskGphMthd method =
        GetMthdForBldSubTskGph(logical_edge->src_node(), logical_edge->dst_node());
    (this->*method)(logical_edge->src_node(), logical_edge->dst_node(),
                    logical2sorted_comp_tasks.at(logical_edge->src_node()),
                    logical2sorted_comp_tasks.at(logical_edge->dst_node()), &logical2sorted_in_box,
                    &logical2sorted_out_box, Mut121BufTask, AllocateCpuThrdId);
  });
  ToDotWithAutoFilePath();
}

#define DEFINE_BLD_SUB_TASK_GRAPH_METHOD(method_name) \
  void TaskGraph::method_name BLD_SUB_TSK_GPH_MTHD_ARGS()

DEFINE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphByBoxing) {
  std::vector<TaskNode*> sorted_out_box_tmp;
  std::vector<TaskNode*>* sorted_out_box = nullptr;
  if (logical2sorted_out_box->find(src_logical) == logical2sorted_out_box->end()) {
    BuildOutBoxing(src_logical, sorted_src_comp_tasks, &((*logical2sorted_out_box)[src_logical]),
                   AllocateCpuThrdId);
  }
  sorted_out_box = &(logical2sorted_out_box->at(src_logical));

  std::vector<TaskNode*> sorted_in_box_tmp;
  std::vector<TaskNode*>* sorted_in_box = nullptr;
  if (logical2sorted_in_box->find(dst_logical) == logical2sorted_in_box->end()) {
    BuildInBoxing(dst_logical, sorted_dst_comp_tasks, &((*logical2sorted_in_box)[dst_logical]),
                  AllocateCpuThrdId);
  }
  sorted_in_box = &(logical2sorted_in_box->at(dst_logical));

  for (TaskNode* src_box : *sorted_out_box) {
    for (TaskNode* dst_box : *sorted_in_box) {
      if (src_box->machine_id() == dst_box->machine_id()) {
        Connect<TaskNode>(src_box, NewEdge(), dst_box);
      } else {
        AddCopyCommNetTask(src_box, dst_box);
      }
    }
  }
}

DEFINE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphByOneToOne) {
  CHECK_EQ(sorted_src_comp_tasks.size(), sorted_dst_comp_tasks.size());
  FOR_RANGE(size_t, i, 0, sorted_src_comp_tasks.size()) {
    CompTaskNode* src = sorted_src_comp_tasks[i];
    CompTaskNode* dst = sorted_dst_comp_tasks[i];
    Connect<TaskNode>(
        Build121BufTo(src, dst->machine_id(), dst->MemZoneId(),
                      [&](int64_t machine_id, int32_t mem_zone_id) {
                        return *Mut121BufTask(src, machine_id, mem_zone_id);
                      },
                      [&](int64_t machine_id, int32_t mem_zone_id, TaskNode* new_val) {
                        TaskNode** cur_val = Mut121BufTask(src, machine_id, mem_zone_id);
                        if (*cur_val == nullptr) {
                          *cur_val = new_val;
                        } else {
                          CHECK_EQ(*cur_val, new_val);
                        }
                        return new_val;
                      }),
        NewEdge(), dst);
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
                         nullptr, Mut121BufTask, AllocateCpuThrdId);
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
    TaskNode* src_d2h_task = AddCopyD2HTaskIfNotCpu(src_comp_task);
    for (CompTaskNode* dst_comp_task : sorted_dst_comp_tasks) {
      if (src_comp_task->parallel_id() == dst_comp_task->parallel_id()) {
        Connect<TaskNode>(src_comp_task, NewEdge(), dst_comp_task);
      } else {
        ConnectWithCopyCommNetIfNeed(src_d2h_task, dst_comp_task);
      }
    }
  }
}

TaskNode* TaskGraph::Build121BufTo(
    TaskNode* src, int64_t dst_machine_id, int32_t dst_mem_zone_id,
    std::function<TaskNode*(int64_t machine_id, int32_t mem_zone_id)> Get121BufTask,
    std::function<TaskNode*(int64_t machine_id, int32_t mem_zone_id, TaskNode*)> Set121BufTask) {
  {
    TaskNode* done = Get121BufTask(dst_machine_id, dst_mem_zone_id);
    if (done) { return done; }
  }
  int32_t cpu_mem_zone_id = Global<IDMgr>::Get()->CpuMemZoneId();

  if (src->machine_id() != dst_machine_id) {
    if (dst_mem_zone_id == cpu_mem_zone_id) {
      TaskNode* src_cpu =
          Build121BufTo(src, src->machine_id(), cpu_mem_zone_id, Get121BufTask, Set121BufTask);
      CopyCommNetTaskNode* copy_comm_net = NewNode<CopyCommNetTaskNode>();
      copy_comm_net->Init(dst_machine_id);
      Connect<TaskNode>(src_cpu, NewEdge(), copy_comm_net);
      return Set121BufTask(dst_machine_id, dst_mem_zone_id, copy_comm_net);
    } else {
      TaskNode* dst_cpu =
          Build121BufTo(src, dst_machine_id, cpu_mem_zone_id, Get121BufTask, Set121BufTask);
      return Build121BufTo(dst_cpu, dst_machine_id, dst_mem_zone_id, Get121BufTask, Set121BufTask);
    }
  } else {
    if (src->MemZoneId() == dst_mem_zone_id) {
      return Set121BufTask(dst_machine_id, dst_mem_zone_id, src);
    } else {
      if (dst_mem_zone_id == cpu_mem_zone_id) {
        return Set121BufTask(dst_machine_id, dst_mem_zone_id, AddCopyD2HTaskIfNotCpu(src));
      } else {
        TaskNode* src_cpu =
            Build121BufTo(src, dst_machine_id, cpu_mem_zone_id, Get121BufTask, Set121BufTask);
        CopyHdTaskNode* src_h2d = NewNode<CopyHdTaskNode>();
        src_h2d->Init(dst_machine_id,
                      Global<IDMgr>::Get()->GetThrdIdFromGpuMemZoneId(dst_mem_zone_id),
                      CopyHdOpConf::H2D);
        Connect<TaskNode>(src_cpu, NewEdge(), src_h2d);
        return Set121BufTask(dst_machine_id, dst_mem_zone_id, src_h2d);
      }
    }
  }
}

TaskNode* TaskGraph::AddCopyH2DTaskIfNotCpu(TaskNode* task) {
  if (task->device_type() == DeviceType::kCPU) { return task; }
  CopyHdTaskNode* copy_task = NewNode<CopyHdTaskNode>();
  copy_task->Init(task->machine_id(), task->thrd_id(), CopyHdOpConf::H2D);
  Connect<TaskNode>(copy_task, NewEdge(), task);
  return copy_task;
}

TaskNode* TaskGraph::AddCopyD2HTaskIfNotCpu(TaskNode* task) {
  if (task->device_type() == DeviceType::kCPU) { return task; }
  CopyHdTaskNode* copy_task = NewNode<CopyHdTaskNode>();
  copy_task->Init(task->machine_id(), task->thrd_id(), CopyHdOpConf::D2H);
  Connect<TaskNode>(task, NewEdge(), copy_task);
  return copy_task;
}

void TaskGraph::AddCopyCommNetTask(TaskNode* src, TaskNode* dst) {
  CHECK_NE(src->machine_id(), dst->machine_id());
  CHECK_EQ(src->device_type(), DeviceType::kCPU);
  CHECK_EQ(dst->device_type(), DeviceType::kCPU);
  CopyCommNetTaskNode* copy_comm_net_task = NewNode<CopyCommNetTaskNode>();
  copy_comm_net_task->Init(dst->machine_id());
  Connect<TaskNode>(src, NewEdge(), copy_comm_net_task);
  Connect<TaskNode>(copy_comm_net_task, NewEdge(), dst);
}

void TaskGraph::BuildOutBoxing(const LogicalNode* logical,
                               const std::vector<CompTaskNode*>& sorted_comp_tasks,
                               std::vector<TaskNode*>* sorted_out_box,
                               std::function<int64_t(const TaskNode*)> AllocateCpuThrdId) {
  std::map<int64_t, std::vector<TaskNode*>> machine_id2bound_task;
  for (CompTaskNode* comp_task : sorted_comp_tasks) {
    TaskNode* task = AddCopyD2HTaskIfNotCpu(comp_task);
    machine_id2bound_task[task->machine_id()].push_back(task);
  }
  for (const auto& pair : machine_id2bound_task) {
    OutBoxingTaskNode* boxing_task = NewNode<OutBoxingTaskNode>();
    boxing_task->set_machine_id(pair.second.front()->machine_id());
    boxing_task->set_thrd_id(AllocateCpuThrdId(boxing_task));
    for (TaskNode* task : pair.second) { Connect<TaskNode>(task, NewEdge(), boxing_task); }
    sorted_out_box->push_back(boxing_task);
  }
}

void TaskGraph::BuildInBoxing(const LogicalNode* logical,
                              const std::vector<CompTaskNode*>& sorted_comp_tasks,
                              std::vector<TaskNode*>* sorted_in_box,
                              std::function<int64_t(const TaskNode*)> AllocateCpuThrdId) {
  std::map<int64_t, std::vector<TaskNode*>> machine_id2bound_task;
  for (CompTaskNode* comp_task : sorted_comp_tasks) {
    TaskNode* task = AddCopyH2DTaskIfNotCpu(comp_task);
    machine_id2bound_task[task->machine_id()].push_back(task);
  }
  for (const auto& pair : machine_id2bound_task) {
    InBoxingTaskNode* boxing_task = NewNode<InBoxingTaskNode>();
    boxing_task->set_machine_id(pair.second.front()->machine_id());
    boxing_task->set_thrd_id(AllocateCpuThrdId(boxing_task));
    for (TaskNode* task : pair.second) { Connect<TaskNode>(boxing_task, NewEdge(), task); }
    sorted_in_box->push_back(boxing_task);
  }
}

void TaskGraph::ConnectWithCopyCommNetIfNeed(TaskNode* src, TaskNode* dst) {
  if (src->machine_id() == dst->machine_id()) {
    Connect(src, NewEdge(), dst);
  } else {
    AddCopyCommNetTask(src, dst);
  }
}

}  // namespace oneflow
