#include "oneflow/core/graph/task_graph.h"
#include "oneflow/core/graph/boxing_task_node.h"
#include "oneflow/core/graph/copy_task_node.h"

namespace oneflow {

TaskGraph::TaskGraph(std::unique_ptr<const ChainGraph>&& chain_gph) {
  chain_gph_ = std::move(chain_gph);
  HashMap<const ChainNode*, std::vector<CompTaskNode*>> chain2sorted_comp_tasks;
  HashMap<const ChainNode*, std::vector<TaskNode*>> chain2sorted_in_box;
  HashMap<const ChainNode*, std::vector<TaskNode*>> chain2sorted_out_box;

  const JobDesc* job_desc = Global<JobDesc>::Get();

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
  chain_gph_->ForEachNode([&](const ChainNode* chain_node) {
    chain_node->GenSortedCompTaskNodes(
        AllocateCpuThrdId, [&](CompTaskNode* comp_task_node) {
          AddAllocatedNode(comp_task_node);
          chain2sorted_comp_tasks[chain_node].push_back(comp_task_node);
        });
  });
  chain_gph_->ForEachEdge([&](const ChainEdge* chain_edge) {
    BldSubTskGphMthd method = chain_edge->GetMthdForBldSubTskGph();
    (this->*method)(chain_edge->src_node(), chain_edge->dst_node(),
                    chain2sorted_comp_tasks.at(chain_edge->src_node()),
                    chain2sorted_comp_tasks.at(chain_edge->dst_node()),
                    &chain2sorted_in_box, &chain2sorted_out_box,
                    AllocateCpuThrdId);
  });
  ToDotWithAutoFilePath();
}

static bool IsDataParallelOneToOneBoxing(const ChainNode* src_chain,
                                         const ChainNode* dst_chain) {
  // TODO: it's ugly
  if (typeid(*src_chain) != typeid(*dst_chain)) { return false; }
  int32_t src_same_type_cnt = 0;
  src_chain->ForEachNodeOnOutEdge([&](const ChainNode* chain) {
    if (typeid(*src_chain) == typeid(*chain)) { src_same_type_cnt += 1; }
  });
  int32_t dst_same_type_cnt = 0;
  dst_chain->ForEachNodeOnInEdge([&](const ChainNode* chain) {
    if (typeid(*dst_chain) == typeid(*chain)) { dst_same_type_cnt += 1; }
  });
  if (src_same_type_cnt > 1) { return false; }
  if (dst_same_type_cnt > 1) { return false; }
  std::shared_ptr<const ParallelDesc> src_desc = src_chain->parallel_desc();
  std::shared_ptr<const ParallelDesc> dst_desc = dst_chain->parallel_desc();
  if (src_desc->policy() != kDataParallel) { return false; }
  if (dst_desc->policy() != kDataParallel) { return false; }
  if (src_desc->parallel_num() != dst_desc->parallel_num()) { return false; }
  return true;
}

#define DEFINE_BLD_SUB_TASK_GRAPH_METHOD(method_name)                          \
  void TaskGraph::method_name(                                                 \
      const ChainNode* src_chain, const ChainNode* dst_chain,                  \
      const std::vector<CompTaskNode*>& sorted_src_comp_tasks,                 \
      const std::vector<CompTaskNode*>& sorted_dst_comp_tasks,                 \
      HashMap<const ChainNode*, std::vector<TaskNode*>>* chain2sorted_in_box,  \
      HashMap<const ChainNode*, std::vector<TaskNode*>>* chain2sorted_out_box, \
      std::function<int64_t(const TaskNode*)> AllocateCpuThrdId)

DEFINE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphByBoxing) {
  if (IsDataParallelOneToOneBoxing(src_chain, dst_chain)) {
    BldSubTskGphByOneToOne(src_chain, dst_chain, sorted_src_comp_tasks,
                           sorted_dst_comp_tasks, nullptr, nullptr,
                           AllocateCpuThrdId);
    return;
  }
  std::vector<TaskNode*> sorted_out_box_tmp;
  std::vector<TaskNode*>* sorted_out_box = nullptr;
  if (src_chain->HasSoleRecurrentOp()) {
    BuildOutBoxing(src_chain, sorted_src_comp_tasks, &sorted_out_box_tmp,
                   AllocateCpuThrdId);
    sorted_out_box = &sorted_out_box_tmp;
  } else {
    if (chain2sorted_out_box->find(src_chain) == chain2sorted_out_box->end()) {
      BuildOutBoxing(src_chain, sorted_src_comp_tasks,
                     &((*chain2sorted_out_box)[src_chain]), AllocateCpuThrdId);
    }
    sorted_out_box = &(chain2sorted_out_box->at(src_chain));
  }

  std::vector<TaskNode*> sorted_in_box_tmp;
  std::vector<TaskNode*>* sorted_in_box = nullptr;
  if (dst_chain->HasSoleRecurrentOp()) {
    BuildInBoxing(dst_chain, sorted_dst_comp_tasks, &sorted_in_box_tmp,
                  AllocateCpuThrdId);
    sorted_in_box = &sorted_in_box_tmp;
  } else {
    if (chain2sorted_in_box->find(dst_chain) == chain2sorted_in_box->end()) {
      BuildInBoxing(dst_chain, sorted_dst_comp_tasks,
                    &((*chain2sorted_in_box)[dst_chain]), AllocateCpuThrdId);
    }
    sorted_in_box = &(chain2sorted_in_box->at(dst_chain));
  }

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
    CompTaskNode* src_comp_task = sorted_src_comp_tasks[i];
    CompTaskNode* dst_comp_task = sorted_dst_comp_tasks[i];
    if (src_comp_task->machine_id() == dst_comp_task->machine_id()) {
      if (src_comp_task->device_type() == dst_comp_task->device_type()) {
        if (src_comp_task->device_type() != DeviceType::kCPU) {
          CHECK_EQ(src_comp_task->thrd_id(), dst_comp_task->thrd_id());
        }
        Connect<TaskNode>(src_comp_task, NewEdge(), dst_comp_task);
      } else {
        CopyHdTaskNode* copy_hd_task = NewNode<CopyHdTaskNode>();
        if (src_comp_task->device_type() == DeviceType::kCPU) {
          copy_hd_task->Init(dst_comp_task, CopyHdOpConf::H2D);
        } else {
          copy_hd_task->Init(src_comp_task, CopyHdOpConf::D2H);
        }
        Connect<TaskNode>(src_comp_task, NewEdge(), copy_hd_task);
        Connect<TaskNode>(copy_hd_task, NewEdge(), dst_comp_task);
      }
    } else {
      TaskNode* src_bound_task = AddCopyD2HTaskIfNotCpu(src_comp_task);
      TaskNode* dst_bound_task = AddCopyH2DTaskIfNotCpu(dst_comp_task);
      AddCopyCommNetTask(src_bound_task, dst_bound_task);
    }
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
  BldSubTskGphByOneToOne(nullptr, nullptr, {selected_src_comp_task},
                         sorted_dst_comp_tasks, nullptr, nullptr,
                         AllocateCpuThrdId);
}

TaskNode* TaskGraph::AddCopyH2DTaskIfNotCpu(CompTaskNode* comp_task) {
  if (comp_task->device_type() == DeviceType::kCPU) { return comp_task; }
  CopyHdTaskNode* copy_task = NewNode<CopyHdTaskNode>();
  copy_task->Init(comp_task, CopyHdOpConf::H2D);
  Connect<TaskNode>(copy_task, NewEdge(), comp_task);
  return copy_task;
}

TaskNode* TaskGraph::AddCopyD2HTaskIfNotCpu(CompTaskNode* comp_task) {
  if (comp_task->device_type() == DeviceType::kCPU) { return comp_task; }
  CopyHdTaskNode* copy_task = NewNode<CopyHdTaskNode>();
  copy_task->Init(comp_task, CopyHdOpConf::D2H);
  Connect<TaskNode>(comp_task, NewEdge(), copy_task);
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

void TaskGraph::BuildOutBoxing(
    const ChainNode* chain, const std::vector<CompTaskNode*>& sorted_comp_tasks,
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
    for (TaskNode* task : pair.second) {
      Connect<TaskNode>(task, NewEdge(), boxing_task);
    }
    sorted_out_box->push_back(boxing_task);
  }
}

void TaskGraph::BuildInBoxing(
    const ChainNode* chain, const std::vector<CompTaskNode*>& sorted_comp_tasks,
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
    for (TaskNode* task : pair.second) {
      Connect<TaskNode>(boxing_task, NewEdge(), task);
    }
    sorted_in_box->push_back(boxing_task);
  }
}

}  // namespace oneflow
