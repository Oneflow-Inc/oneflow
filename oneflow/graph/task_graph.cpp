#include "graph/task_graph.h"
#include "graph/comp_task_node.h"
#include "graph/comm_net_task_node.h"
#include "graph/in_boxing_task_node.h"
#include "graph/out_boxing_task_node.h"

namespace oneflow {

namespace {

inline void TaskConnect(TaskNode* src_node,
                        TaskEdge* edge,
                        TaskNode* dst_node) {
  Connect<TaskNode, TaskEdge>(src_node, edge, dst_node);
}

}

void TaskGraph::BuildExecAndProducedRegsts() {
  for (TaskNode& node : *this) {
    node.BuildExecAndProducedRegsts(this);
  }
}

std::vector<CompTaskNode*> TaskGraph::SortedCompTasksInChain(
    const ChainNode* chain) const {
  std::vector<CompTaskNode*> ret;
  for (const auto& node : nodes()) {
    auto comp_node = dynamic_cast<CompTaskNode*> (node.get());
    if (comp_node && comp_node->chain_node() == chain) {
      ret.push_back(comp_node);
    }
  }
  return ret;
}

void TaskGraph::BuildFromChainGph(
    std::unique_ptr<ChainGraph>&& chain_gph,
    bool need_bp) {
  stage_gph_.reset(new StageGraph(std::move(chain_gph)));
  BuildFromStageGph(need_bp);
}

void TaskGraph::BuildFromStageGph(bool need_bp) {
  LOG(INFO) << "Build FwTaskGraph...";
  Stage2TaskNodesMap stage2task_nodes;
  InitCompTaskNodes(&stage2task_nodes);
  InitBoxingTaskNodes(&stage2task_nodes);
  ConnectBoxingTaskNodes(&stage2task_nodes);
  UpdateSourceAndSink();
  ToDotFile(LogDir() + "/fw_task_graph.dot");
  if (need_bp) {
    BuildBpStruct();
  }
}

void TaskGraph::InitCompTaskNodes(Stage2TaskNodesMap* stage2task_nodes) {
  for (const std::unique_ptr<StageNode>& stage : stage_gph_->nodes()) {
    if (stage->chain_node()->parallel_desc()->device_type() == kGPU) {
      Stage2DeviceCompTaskNodes(stage.get(),
                                &((*stage2task_nodes)[stage.get()]));
    } else {
      Stage2HostCompTaskNodes(stage.get(),
                              &((*stage2task_nodes)[stage.get()]));
    }
  }
}

void TaskGraph::Stage2DeviceCompTaskNodes(
    const StageNode* stage,
    TaskNodesInStage* task_nodes_in_stage) {
  uint64_t parallel_idx = stage->parallel_range().begin();
  for (auto device_phy_id : stage->SortedDevicePhyIds()) {
    uint64_t thread_local_id =
        IDMgr::Singleton().ThrdLocId4DevicePhyId(device_phy_id);
    // comp_task_node
    DeviceCompTaskNode* comp_task_node = NewTaskNode<DeviceCompTaskNode> ();
    comp_task_node->SetFwNode();
    comp_task_node->set_stage_node(stage);
    comp_task_node->mut_thrd_loc_id() = thread_local_id;
    comp_task_node->set_parallel_id(parallel_idx++);
    comp_task_node->set_task_id();
    // comp_in_task_node
    if (!stage->in_edges().empty()) {
      CopyHDTaskNode* comp_in_task_node = NewTaskNode<CopyHDTaskNode> ();
      comp_in_task_node->SetFwNode();
      comp_in_task_node->set_stage_node(stage);
      comp_in_task_node->mut_thrd_loc_id() = thread_local_id;
      comp_in_task_node->SetFwInCopy();
      comp_in_task_node->set_task_id();
      TaskConnect(comp_in_task_node, NewEdge(), comp_task_node);
      task_nodes_in_stage->comp_in_task_nodes.push_back(comp_in_task_node);
    } else {
      task_nodes_in_stage->comp_in_task_nodes.push_back(comp_task_node);
    }
    // comp_out_task_node
    if (!stage->out_edges().empty()) {
      CopyHDTaskNode* comp_out_task_node = NewTaskNode<CopyHDTaskNode> ();
      comp_out_task_node->SetFwNode();
      comp_out_task_node->set_stage_node(stage);
      comp_out_task_node->mut_thrd_loc_id() = thread_local_id;
      comp_out_task_node->SetFwOutCopy();
      comp_out_task_node->set_task_id();
      TaskConnect(comp_task_node, NewEdge(), comp_out_task_node);
      task_nodes_in_stage->comp_out_task_nodes.push_back(comp_out_task_node);
    } else {
      task_nodes_in_stage->comp_out_task_nodes.push_back(comp_task_node);
    }
  }
  CHECK_EQ(parallel_idx, stage->parallel_range().end());
}

void TaskGraph::Stage2HostCompTaskNodes(const StageNode* stage,
                                        TaskNodesInStage* task_nodes_in_stage) {
  uint64_t parallel_begin = stage->parallel_range().begin();
  uint64_t parallel_end = stage->parallel_range().end();
  uint64_t parallel_idx = parallel_begin;
  while (parallel_idx < parallel_end) {
    HostCompTaskNode* comp_task_node = NewTaskNode<HostCompTaskNode> ();
    comp_task_node->SetFwNode();
    comp_task_node->set_stage_node(stage);
    comp_task_node->set_parallel_id(parallel_idx++);
    comp_task_node->set_task_id();
    // Set comp_task_node::thread_local_id
    if (stage->SortedDevicePhyIds().empty()) {
      comp_task_node->mut_thrd_loc_id() = IDMgr::Singleton().DiskThrdLocId();
    } else {
      auto device_id = stage->SortedDevicePhyIds().at(parallel_idx - parallel_begin);
      comp_task_node->mut_thrd_loc_id() =
          IDMgr::Singleton().ThrdLocId4DevicePhyId(device_id);
    }
    // 
    task_nodes_in_stage->comp_in_task_nodes.push_back(comp_task_node);
    task_nodes_in_stage->comp_out_task_nodes.push_back(comp_task_node);
  }
}

void TaskGraph::InitBoxingTaskNodes(Stage2TaskNodesMap* stage2task_nodes) {
  for (const std::unique_ptr<StageNode>& stage : stage_gph_->nodes()) {
    InitInboxingTaskNode(stage.get(), &(stage2task_nodes->at(stage.get())));
    InitOutBoxingTaskNode(stage.get(), &(stage2task_nodes->at(stage.get())));
  }
}

void TaskGraph::InitInboxingTaskNode(const StageNode* stage,
                                     TaskNodesInStage* task_nodes_in_stage) {
  task_nodes_in_stage->in_boxing_task_node = nullptr;
  if (stage->in_edges().size() == 1
      && task_nodes_in_stage->comp_in_task_nodes.size() == 1) {
    return;
  }
  InBoxingTaskNode* boxing_task = NewTaskNode<InBoxingTaskNode> ();
  boxing_task->SetFwNode();
  boxing_task->set_stage_node(stage);
  boxing_task->mut_thrd_loc_id() = IDMgr::Singleton().BoxingThrdLocId();
  boxing_task->set_task_id();
  for (TaskNode* comp_in_task : task_nodes_in_stage->comp_in_task_nodes) {
    TaskConnect(boxing_task, NewEdge(), comp_in_task);
  }
  task_nodes_in_stage->in_boxing_task_node = boxing_task;
}

void TaskGraph::InitOutBoxingTaskNode(
    const StageNode* stage,
    TaskNodesInStage* task_nodes_in_stage) {
  task_nodes_in_stage->out_boxing_task_node = nullptr;
  if (stage->out_edges().size() == 1
      && task_nodes_in_stage->comp_out_task_nodes.size() == 1) {
    return;
  }
  OutBoxingTaskNode* boxing_task = NewTaskNode<OutBoxingTaskNode> ();
  boxing_task->SetFwNode();
  boxing_task->set_stage_node(stage);
  boxing_task->mut_thrd_loc_id() = IDMgr::Singleton().BoxingThrdLocId();
  boxing_task->set_task_id();
  for (TaskNode* comp_out_task : task_nodes_in_stage->comp_out_task_nodes) {
    TaskConnect(comp_out_task, NewEdge(), boxing_task);
  }
  task_nodes_in_stage->out_boxing_task_node = boxing_task;
}

void TaskGraph::ConnectBoxingTaskNodes(
    const Stage2TaskNodesMap* stage2task_nodes) {
  for (const std::unique_ptr<StageNode>& cur_stage : stage_gph_->nodes()) {
    const TaskNodesInStage& cur_tasks = stage2task_nodes->at(cur_stage.get());
    TaskNode* out_node = cur_tasks.out_boxing_task_node;
    if (out_node == nullptr) {
      CHECK_EQ(cur_tasks.comp_out_task_nodes.size(), 1);
      out_node = cur_tasks.comp_out_task_nodes[0];
    }
    for (const StageEdge* edge : cur_stage->out_edges()) {
      StageNode* succ_stage = edge->dst_node();
      const TaskNodesInStage& succ_tasks = stage2task_nodes->at(succ_stage);
      TaskNode* in_node = succ_tasks.in_boxing_task_node;
      if (in_node == nullptr) {
        CHECK_EQ(succ_tasks.comp_in_task_nodes.size(), 1);
        in_node = succ_tasks.comp_in_task_nodes[0];
      }
      if (cur_stage->machine_id() == succ_stage->machine_id()) {
        TaskConnect(out_node, NewEdge(), in_node);
        continue;
      }
      CommNetTaskNode* out_comm_net_node = NewTaskNode<CommNetTaskNode> ();
      out_comm_net_node->SetFwNode();
      out_comm_net_node->set_stage_node(cur_stage.get());
      out_comm_net_node->mut_thrd_loc_id() =
          IDMgr::Singleton().CommNetThrdLocId();
      out_comm_net_node->SetFwSender();
      out_comm_net_node->set_task_id();
      CommNetTaskNode* in_comm_net_node = NewTaskNode<CommNetTaskNode> ();
      in_comm_net_node->SetFwNode();
      in_comm_net_node->set_stage_node(succ_stage);
      in_comm_net_node->mut_thrd_loc_id() =
          IDMgr::Singleton().CommNetThrdLocId();
      in_comm_net_node->SetFwReceiver();
      in_comm_net_node->set_task_id();
      TaskConnect(out_node, NewEdge(), out_comm_net_node);
      TaskConnect(out_comm_net_node, NewEdge(), in_comm_net_node);
      TaskConnect(in_comm_net_node, NewEdge(), in_node);
    }
  }
}

void TaskGraph::BuildBpStruct() {
  LOG(INFO) << "Build BpTaskGraph...";
  std::vector<TaskNode*> turning_node_vec;
  GenerateRelatedBpNodes(&turning_node_vec);
  BackwardConnect(turning_node_vec);
  UpdateSourceAndSink();
}

void TaskGraph::GenerateRelatedBpNodes(
    std::vector<TaskNode*> *turning_node_vec) {
  for (auto task_node = begin(); task_node != end(); ++task_node) {
    if (auto comp_task_node = dynamic_cast<CompTaskNode*> (&(*task_node))) {
      if (!comp_task_node->IsLossNode()) {
        comp_task_node->BuildAndConnectBpNode();
      } else {
        turning_node_vec->push_back(&(*task_node));
      }
    } else {
      for (TaskEdge* edge : task_node->in_edges()) {
        if (edge->src_node()->GetBpNode() != nullptr) {
          task_node->BuildAndConnectBpNode();
          break;
        }
      }
    }
  }
}

void TaskGraph::BackwardConnect(
    const std::vector<TaskNode*>& turning_node_vec) {
  std::queue<TaskNode*> bp_node_queue;
  for (TaskNode* turning_node : turning_node_vec) {
    for (TaskEdge* fw_edge : turning_node->in_edges()) {
      TaskNode* bp_pred_node = fw_edge->src_node()->GetBpNode();
      TaskEdge* bp_edge = NewEdge();
      TaskConnect(turning_node, bp_edge, bp_pred_node);
      fw_edge->set_related_fwbp_edge(bp_edge);
      bp_edge->set_related_fwbp_edge(fw_edge);
      bp_node_queue.push(bp_pred_node);
    }
  }
  while (!bp_node_queue.empty()) {
    TaskNode* bp_cur_node = bp_node_queue.front();
    bp_node_queue.pop();
    for (TaskEdge* fw_edge : bp_cur_node->GetFwNode()->in_edges()) {
      TaskNode* bp_pred_node = fw_edge->src_node()->GetBpNode();
      TaskEdge* bp_edge = NewEdge();
      fw_edge->set_related_fwbp_edge(bp_edge);
      bp_edge->set_related_fwbp_edge(fw_edge);
      TaskConnect(bp_cur_node, bp_edge, bp_pred_node);
      bp_node_queue.push(bp_pred_node);
    }
  }
}

} // namespace oneflow
