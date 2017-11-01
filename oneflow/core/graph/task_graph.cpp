#include "oneflow/core/graph/task_graph.h"
#include "oneflow/core/graph/boxing_task_node.h"
#include "oneflow/core/graph/copy_task_node.h"

namespace oneflow {

TaskGraph::TaskGraph(std::unique_ptr<const ChainGraph>&& chain_gph) {
  chain_gph_ = std::move(chain_gph);
  BuildStruct();
  TODO();
  ForEachNode(std::bind(&TaskNode::ProduceAllRegstsAndBindEdges,
                        std::placeholders::_1));
  ForEachNode(std::bind(&TaskNode::ConsumeAllRegsts, std::placeholders::_1));
  ForEachNode(std::bind(&TaskNode::Build, std::placeholders::_1),
              std::bind(&TaskNode::IsReadyForBuild, std::placeholders::_1));
}

void TaskGraph::BldSubTskGphByBoxing(
    const ChainNode* src_chain, const ChainNode* dst_chain,
    const std::vector<CompTaskNode*>& sorted_src_comp_tasks,
    const std::vector<CompTaskNode*>& sorted_dst_comp_tasks,
    HashMap<const ChainNode*, std::vector<TaskNode*>>* chain2sorted_in_box,
    HashMap<const ChainNode*, std::vector<TaskNode*>>* chain2sorted_out_box) {
  BuildOutBoxingIfNeed(src_chain, sorted_src_comp_tasks, chain2sorted_out_box);
  BuildInBoxingIfNeed(dst_chain, sorted_dst_comp_tasks, chain2sorted_in_box);
  for (TaskNode* src_box : chain2sorted_out_box->at(src_chain)) {
    for (TaskNode* dst_box : chain2sorted_in_box->at(dst_chain)) {
      if (src_box->machine_id() == dst_box->machine_id()) {
        Connect<TaskNode>(src_box, NewEdge(), dst_box);
      } else {
        AddCopyCommNetTask(src_box, dst_box);
      }
    }
  }
}

void TaskGraph::BldSubTskGphByOneToOne(
    const ChainNode* src_chain, const ChainNode* dst_chain,
    const std::vector<CompTaskNode*>& sorted_src_comp_tasks,
    const std::vector<CompTaskNode*>& sorted_dst_comp_tasks,
    HashMap<const ChainNode*, std::vector<TaskNode*>>*,
    HashMap<const ChainNode*, std::vector<TaskNode*>>*) {
  CHECK_EQ(sorted_src_comp_tasks.size(), sorted_dst_comp_tasks.size());
  FOR_RANGE(size_t, i, 0, sorted_src_comp_tasks.size()) {
    CompTaskNode* src_comp_task = sorted_src_comp_tasks[i];
    CompTaskNode* dst_comp_task = sorted_dst_comp_tasks[i];
    if (src_comp_task->machine_id() == dst_comp_task->machine_id()) {
      if (src_comp_task->device_type() == dst_comp_task->device_type()) {
        if (src_comp_task->device_type() != DeviceType::kCPU) {
          CHECK_EQ(src_comp_task->thrd_loc_id(), dst_comp_task->thrd_loc_id());
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

void TaskGraph::BldSubTskGphBySelectOneSourceToSoleSink(
    const ChainNode* src_chain, const ChainNode* dst_chain,
    const std::vector<CompTaskNode*>& sorted_src_comp_tasks,
    const std::vector<CompTaskNode*>& sorted_dst_comp_tasks,
    HashMap<const ChainNode*, std::vector<TaskNode*>>*,
    HashMap<const ChainNode*, std::vector<TaskNode*>>*) {
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
      if (src_comp_task->thrd_loc_id() == sole_dst_comp_task->thrd_loc_id()) {
        UpdateSelected(src_comp_task);
        break;
      }
    }
  }
  CHECK_NOTNULL(selected_src_comp_task);
  BldSubTskGphByOneToOne(nullptr, nullptr, {selected_src_comp_task},
                         sorted_dst_comp_tasks, nullptr, nullptr);
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

void TaskGraph::BuildOutBoxingIfNeed(
    const ChainNode* chain, const std::vector<CompTaskNode*>& sorted_comp_tasks,
    HashMap<const ChainNode*, std::vector<TaskNode*>>* chain2sorted_out_box) {
  if (chain2sorted_out_box->find(chain) != chain2sorted_out_box->end()) {
    return;
  }
  std::map<int64_t, std::vector<TaskNode*>> machine_id2bound_task;
  for (CompTaskNode* comp_task : sorted_comp_tasks) {
    TaskNode* task = AddCopyD2HTaskIfNotCpu(comp_task);
    machine_id2bound_task[task->machine_id()].push_back(task);
  }
  for (const auto& pair : machine_id2bound_task) {
    OutBoxingTaskNode* boxing_task = NewNode<OutBoxingTaskNode>();
    boxing_task->Init(pair.second.front()->machine_id());
    for (TaskNode* task : pair.second) {
      Connect<TaskNode>(task, NewEdge(), boxing_task);
    }
    (*chain2sorted_out_box)[chain].push_back(boxing_task);
  }
}

void TaskGraph::BuildInBoxingIfNeed(
    const ChainNode* chain, const std::vector<CompTaskNode*>& sorted_comp_tasks,
    HashMap<const ChainNode*, std::vector<TaskNode*>>* chain2sorted_in_box) {
  if (chain2sorted_in_box->find(chain) != chain2sorted_in_box->end()) {
    return;
  }
  std::map<int64_t, std::vector<TaskNode*>> machine_id2bound_task;
  for (CompTaskNode* comp_task : sorted_comp_tasks) {
    TaskNode* task = AddCopyH2DTaskIfNotCpu(comp_task);
    machine_id2bound_task[task->machine_id()].push_back(task);
  }
  for (const auto& pair : machine_id2bound_task) {
    InBoxingTaskNode* boxing_task = NewNode<InBoxingTaskNode>();
    boxing_task->Init(pair.second.front()->machine_id());
    for (TaskNode* task : pair.second) {
      Connect<TaskNode>(boxing_task, NewEdge(), task);
    }
    (*chain2sorted_in_box)[chain].push_back(boxing_task);
  }
}

void TaskGraph::BuildStruct() {
  HashMap<const ChainNode*, std::vector<CompTaskNode*>> chain2sorted_comp_tasks;
  HashMap<const ChainNode*, std::vector<TaskNode*>> chain2sorted_in_box;
  HashMap<const ChainNode*, std::vector<TaskNode*>> chain2sorted_out_box;
  chain_gph_->ForEachNode([&](const ChainNode* chain_node) {
    chain_node->GenSortedCompTaskNodes([&](CompTaskNode* comp_task_node) {
      comp_task_node->FixThrdLocId();
      AddAllocatedNode(comp_task_node);
      chain2sorted_comp_tasks[chain_node].push_back(comp_task_node);
    });
  });
  chain_gph_->ForEachEdge([&](const ChainEdge* chain_edge) {
    BldSubTskGphMthd method = chain_edge->GetMthdForBldSubTskGph();
    (this->*method)(chain_edge->src_node(), chain_edge->dst_node(),
                    chain2sorted_comp_tasks.at(chain_edge->src_node()),
                    chain2sorted_comp_tasks.at(chain_edge->dst_node()),
                    &chain2sorted_in_box, &chain2sorted_out_box);
  });
  // TODO: delete boxing with sole in and sole out
  ToDotWithAutoFilePath();
}

}  // namespace oneflow
