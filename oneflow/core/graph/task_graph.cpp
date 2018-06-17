#include "oneflow/core/graph/task_graph.h"
#include "oneflow/core/graph/boxing_task_node.h"
#include "oneflow/core/graph/copy_task_node.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

namespace {
struct Chain {
  // nodes belong to this chain
  std::vector<TaskNode*> nodes;
  // ancestors of the nodes in this chain
  HashSet<TaskNode*> ancestors;
  // ancestors_and_this = nodes + ancestors
  HashSet<TaskNode*> ancestors_and_this;
  int64_t stream_id;
  int64_t path_id;
};

using ChainIt = std::list<Chain>::iterator;
using Task2ChainItMap = HashMap<const TaskNode*, ChainIt>;

void InitChains(const TaskGraph& task_graph, std::list<Chain>* chain_list,
                Task2ChainItMap* task2chain_it) {
  chain_list->clear();
  task2chain_it->clear();
  for (const auto& task_node : task_graph.ordered_task_nodes()) {
    chain_list->emplace_back();
    task2chain_it->insert({task_node, --chain_list->end()});
    Chain& cur_chain = chain_list->back();
    cur_chain.nodes = {task_node};
    cur_chain.path_id = static_cast<int64_t>(task_node->GetPathType());
    cur_chain.stream_id = task_node->GlobalWorkStreamId();
    cur_chain.ancestors.clear();
    cur_chain.ancestors_and_this.clear();
    cur_chain.ancestors_and_this.insert(cur_chain.nodes.begin(), cur_chain.nodes.end());
    cur_chain.ancestors.insert(task_node->ancestors().begin(), task_node->ancestors().end());
    cur_chain.ancestors_and_this.insert(cur_chain.ancestors.begin(), cur_chain.ancestors.end());
  }
}

bool DoMergeWithConnect(std::list<ChainIt>& chains, ChainIt rhs, Task2ChainItMap* task2chain_it) {
  for (auto chains_it = chains.rbegin(); chains_it != chains.rend(); ++chains_it) {
    ChainIt lhs = *chains_it;
    if (lhs->ancestors_and_this == rhs->ancestors) {
      for (TaskNode* node : rhs->nodes) {
        lhs->nodes.push_back(node);
        lhs->ancestors_and_this.insert(node);
        task2chain_it->at(node) = rhs;
      }
      return true;
    }
  }
  return false;
}

bool DoMergeWithoutConnect(std::list<ChainIt>& chains, ChainIt rhs,
                           Task2ChainItMap* task2chain_it) {
  for (auto chains_it = chains.rbegin(); chains_it != chains.rend(); ++chains_it) {
    ChainIt lhs = *chains_it;
    if (lhs->ancestors == rhs->ancestors) {
      for (TaskNode* node : rhs->nodes) {
        lhs->nodes.push_back(node);
        lhs->ancestors_and_this.insert(node);
        task2chain_it->at(node) = lhs;
      }
      return true;
    }
  }
  return false;
}

bool TryMerge(
    std::list<Chain>* chain_list, Task2ChainItMap* task2chain_it,
    std::function<bool(std::list<ChainIt>& chains, ChainIt cur_it, Task2ChainItMap* task2chain_it)>
        DoMerge) {
  HashMap<std::pair<int64_t, int64_t>, std::list<ChainIt>, pair_hash> stream_path2chains;
  bool merge_happened = false;
  for (auto cur_chain_it = chain_list->begin(); cur_chain_it != chain_list->end();) {
    std::pair<int64_t, int64_t> stream_path_id = {cur_chain_it->stream_id, cur_chain_it->path_id};
    auto stream_path_it = stream_path2chains.find(stream_path_id);
    if (stream_path_it == stream_path2chains.end()) {
      CHECK(stream_path2chains
                .insert({{cur_chain_it->stream_id, cur_chain_it->path_id}, {cur_chain_it}})
                .second);
      ++cur_chain_it;
    } else {
      if (DoMerge(stream_path_it->second, cur_chain_it, task2chain_it)) {
        cur_chain_it = chain_list->erase(cur_chain_it);
        merge_happened = true;
      } else {
        stream_path2chains[stream_path_id].push_back(cur_chain_it);
        ++cur_chain_it;
      }
    }
  }
  return merge_happened;
}

void MergeChains(std::list<Chain>* chain_list, Task2ChainItMap* task2chain_it) {
  while (TryMerge(chain_list, task2chain_it, DoMergeWithConnect)
         || TryMerge(chain_list, task2chain_it, DoMergeWithoutConnect)) {}
}

}  // namespace

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
          comp_task_node->SetPathType(logical_node->GetPathType());
        });
  });
  logical_gph_->ForEachEdge([&](const LogicalEdge* logical_edge) {
    BldSubTskGphMthd method =
        GetMthdForBldSubTskGph(logical_edge->src_node(), logical_edge->dst_node());
    (this->*method)(logical_edge->src_node(), logical_edge->dst_node(),
                    logical2sorted_comp_tasks.at(logical_edge->src_node()),
                    logical2sorted_comp_tasks.at(logical_edge->dst_node()), &logical2sorted_in_box,
                    &logical2sorted_out_box, Mut121BufTask, AllocateCpuThrdIdEvenly);
    SetPathTypeForNewNodes(logical_edge->src_node(), logical_edge->dst_node());
  });
  ToDotWithAutoFilePath();
}

void TaskGraph::CollectTaskNodesInSameType() {
  std::map<PathType, HashSet<TaskNode*>> path_type2task_nodes;
  std::map<int64_t, HashSet<TaskNode*>> stream_id2task_nodes;
  ForEachNode([&](TaskNode* node) {
    CHECK(path_type2task_nodes[node->GetPathType()].insert(node).second);
    CHECK(stream_id2task_nodes[node->GlobalWorkStreamId()].insert(node).second);
  });
  // LOG(INFO) << "path_type2task_nodes";
  for (const auto& pair : path_type2task_nodes) {
    LOG(INFO) << pair.first << ":" << pair.second.size();
  }
  // LOG(INFO) << "stream_id2task_nodes";
  for (const auto& pair : stream_id2task_nodes) {
    LOG(INFO) << pair.first << ":" << pair.second.size();
  }
}

void TaskGraph::OrderAllTaskNodes() {
  UncyclicTopoForEachNode([this](TaskNode* node) { ordered_task_nodes_.emplace_back(node); });
}

void TaskGraph::FindChainsInSameStream() {
  HashMap<int64_t, HashSet<TaskNode*>> chain_id2task_nodes;
  OrderAllTaskNodes();
  CollectAncestorsForEachTaskNode();

  std::list<Chain> chain_list;
  Task2ChainItMap task2chain_it;
  InitChains(*this, &chain_list, &task2chain_it);
  MergeChains(&chain_list, &task2chain_it);

  for (auto& chain : chain_list) {
    int64_t chain_id =
        Global<IDMgr>::Get()->AllocateChainId(chain.nodes.front()->GlobalWorkStreamId());
    for (auto task_node : chain.nodes) {
      task_node->set_chain_id(chain_id);
      CHECK(chain_id2task_nodes[chain_id].insert(task_node).second);
    }
  }
}

void TaskGraph::AddOrderCtrlEdgeInSameChain() {
  HashMap<int64_t, TaskNode*> chain_id2node;
  UncyclicTopoForEachNode([&](TaskNode* node) {
    int64_t chain_id = node->chain_id();
    auto iter = chain_id2node.find(chain_id);
    if (iter == chain_id2node.end()) {
      CHECK(chain_id2node.emplace(chain_id, node).second);
    } else {
      iter->second->BuildDelayRegstDescIfNeed(node);
      iter->second = node;
    }
  });
}

void TaskGraph::AddMutexCtrlEdgeInSameChain() {
  // TODO
}

void TaskGraph::AddOrderCtrlEdgeBetweenCopyAndMdUpdt() {
  // TODO
}

void TaskGraph::CollectAncestorsForEachTaskNode() {
  for (TaskNode* task_node : ordered_task_nodes_) {
    task_node->mut_ancestors().clear();
    task_node->ForEachNodeOnInEdge([&](TaskNode* node_on_in_edge) {
      if (node_on_in_edge->GetTaskType() != TaskType::kNormalMdUpdt) {
        task_node->mut_ancestors().insert(node_on_in_edge->ancestors().begin(),
                                          node_on_in_edge->ancestors().end());
      }
    });
  }
}

void TaskGraph::UncyclicTopoForEachNode(std::function<void(TaskNode* node)> handler) {
  std::list<TaskNode*> starts;
  ForEachNode([&](TaskNode* node) {
    if (node->consumed_regsts().empty() && !node->IsMeaningLess()) { starts.push_back(node); }
  });
  auto ForEachInNode = [&](TaskNode* node, const std::function<void(TaskNode*)>& handler) {
    node->ForEachNodeOnInEdge([&](TaskNode* node_on_in_edge) {
      if (node_on_in_edge->GetTaskType() != TaskType::kNormalMdUpdt) {
        handler(const_cast<TaskNode*>(node_on_in_edge));
      }
    });
  };
  auto ForEachOutNode = [&](TaskNode* node, const std::function<void(TaskNode*)>& handler) {
    if (node->GetTaskType() != TaskType::kNormalMdUpdt) {
      node->ForEachNodeOnOutEdge(
          [&](TaskNode* node_on_out_edge) { handler(const_cast<TaskNode*>(node_on_out_edge)); });
    }
  };
  TopoForEachNode(starts, ForEachInNode, ForEachOutNode, handler);
}

void TaskGraph::SetPathTypeForNewNodes(const LogicalNode* src_logical,
                                       const LogicalNode* dst_logical) {
  CHECK(src_logical != nullptr && dst_logical != nullptr);
  ForEachNode([&](TaskNode* node) {
    if (node->GetPathType() != kInvalidPath) return;
    if (src_logical->GetPathType() == dst_logical->GetPathType()) {
      node->SetPathType(src_logical->GetPathType());
    } else {
      node->SetPathType(kBoundaryPath);
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
        Build121BufTo(src, dst->machine_id(), dst->MemZoneId121(),
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
      copy_comm_net->Init(dst_machine_id, src_cpu->machine_id());
      Connect<TaskNode>(src_cpu, NewEdge(), copy_comm_net);
      return Set121BufTask(dst_machine_id, dst_mem_zone_id, copy_comm_net);
    } else {
      TaskNode* dst_cpu =
          Build121BufTo(src, dst_machine_id, cpu_mem_zone_id, Get121BufTask, Set121BufTask);
      return Build121BufTo(dst_cpu, dst_machine_id, dst_mem_zone_id, Get121BufTask, Set121BufTask);
    }
  } else {
    if (src->MemZoneId121() == dst_mem_zone_id) {
      return Set121BufTask(dst_machine_id, dst_mem_zone_id, src);
    } else {
      if (dst_mem_zone_id == cpu_mem_zone_id) {
        return Set121BufTask(dst_machine_id, dst_mem_zone_id, AddCopyD2HTaskIfNotCpu(src));
      } else {
        TaskNode* src_cpu =
            Build121BufTo(src, dst_machine_id, cpu_mem_zone_id, Get121BufTask, Set121BufTask);
        CopyHdTaskNode* src_h2d = NewNode<CopyHdTaskNode>();
        src_h2d->Init(CopyHdOpConf::H2D, dst_machine_id,
                      Global<IDMgr>::Get()->GetGpuPhyIdFromMemZoneId(dst_mem_zone_id));
        Connect<TaskNode>(src_cpu, NewEdge(), src_h2d);
        return Set121BufTask(dst_machine_id, dst_mem_zone_id, src_h2d);
      }
    }
  }
}

TaskNode* TaskGraph::AddCopyH2DTaskIfNotCpu(TaskNode* task) {
  if (task->device_type() == DeviceType::kCPU) { return task; }
  CopyHdTaskNode* copy_task = NewNode<CopyHdTaskNode>();
  copy_task->Init(CopyHdOpConf::H2D, task->machine_id(), task->GpuPhyId());
  Connect<TaskNode>(copy_task, NewEdge(), task);
  return copy_task;
}

TaskNode* TaskGraph::AddCopyD2HTaskIfNotCpu(TaskNode* task) {
  if (task->device_type() == DeviceType::kCPU) { return task; }
  CopyHdTaskNode* copy_task = NewNode<CopyHdTaskNode>();
  copy_task->Init(CopyHdOpConf::D2H, task->machine_id(), task->GpuPhyId());
  Connect<TaskNode>(task, NewEdge(), copy_task);
  return copy_task;
}

void TaskGraph::AddCopyCommNetTask(TaskNode* src, TaskNode* dst) {
  CHECK_NE(src->machine_id(), dst->machine_id());
  CopyCommNetTaskNode* copy_comm_net_task = NewNode<CopyCommNetTaskNode>();
  copy_comm_net_task->Init(dst->machine_id(), src->machine_id());
  Connect<TaskNode>(src, NewEdge(), copy_comm_net_task);
  Connect<TaskNode>(copy_comm_net_task, NewEdge(), dst);
}

void TaskGraph::BuildOutBoxing(const LogicalNode* logical,
                               const std::vector<CompTaskNode*>& sorted_comp_tasks,
                               std::vector<TaskNode*>* sorted_out_box,
                               std::function<int64_t(const TaskNode*)> AllocateCpuThrdIdEvenly) {
  std::map<int64_t, std::vector<TaskNode*>> machine_id2bound_task;
  for (CompTaskNode* comp_task : sorted_comp_tasks) {
    TaskNode* task = AddCopyD2HTaskIfNotCpu(comp_task);
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
    TaskNode* task = AddCopyH2DTaskIfNotCpu(comp_task);
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
    AddCopyCommNetTask(src, dst);
  }
}

}  // namespace oneflow
