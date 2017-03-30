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

void TaskGraph::Init(const DLNetConf& dl_net_conf,
                     const Strategy& strategy_conf,
                     bool need_bp) {
  std::unique_ptr<LogicalGraph> logical_graph(new LogicalGraph);
  logical_graph->Init(dl_net_conf, strategy_conf);
  std::unique_ptr<ChainGraph> chain_graph(new ChainGraph);
  chain_graph->Init(logical_graph.get());
  auto stage_graph_raw_ptr = new StageGraph;
  stage_graph_raw_ptr->Init(std::move(chain_graph));
  stage_graph_.reset(stage_graph_raw_ptr);
  BuildWithoutExecGraph(need_bp);
  BuildExecGraph();
}

void TaskGraph::BuildWithoutExecGraph(
    bool need_bp) {
  Stage2TaskNodesMap stage2task_nodes;
  InitCompTaskNodes(&stage2task_nodes);
  InitBoxingTaskNodes(&stage2task_nodes);
  ConnectTaskNodes(&stage2task_nodes);
  UpdateSourceAndSink();
  if (need_bp) {
    BuildBpStruct();
  }
}

void TaskGraph::BuildExecGraph() {
  for (TaskNode& task_node : *this) {
    task_node.BuildExecGraphAndSetRegisterDescs();
  }
}

void TaskGraph::InitCompTaskNodes(Stage2TaskNodesMap* stage2task_nodes) {
  for (const std::unique_ptr<StageNode>& stage : stage_graph_->nodes()) {
    bool is_first_stage = stage_graph_->IsFirstNode(stage.get());
    bool is_last_stage = stage_graph_->IsLastNode(stage.get());
    if (stage->chain_node()->parallel_desc()->engine()
            == ParallelDesc::Engine::kDevice) {
      Stage2DeviceCompTaskNodes(stage.get(),
                                &((*stage2task_nodes)[stage.get()]),
                                is_first_stage,
                                is_last_stage);
    } else {
      Stage2HostCompTaskNodes(stage.get(),
                              &((*stage2task_nodes)[stage.get()]));
    }
  }
}

void TaskGraph::Stage2DeviceCompTaskNodes(
    const StageNode* stage,
    TaskNodesInStage* task_nodes_in_stage,
    bool is_first_stage,
    bool is_last_stage) {
  MachineId machine_id = stage->machine_id();
  int32_t parallel_id = stage->parallel_range().begin;
  for (auto device_physical_id :
      stage->chain_node()->parallel_desc()->sorted_devices_on_machine(machine_id)) {
    ThreadLocalId thread_local_id =
        IDManager::Singleton().ThreadLocalIdFromDevicePhysicalId(device_physical_id);
    // comp_task_node
    DeviceCompTaskNode* comp_task_node = NewTaskNode<DeviceCompTaskNode> ();
    comp_task_node->set_stage_node(stage);
    comp_task_node->mut_thread_local_id() = thread_local_id;
    comp_task_node->SetFwNode();
    comp_task_node->set_parallel_id(parallel_id++);
    // comp_in_task_node
    if (!is_first_stage) {
      CopyHDTaskNode* comp_in_task_node = NewTaskNode<CopyHDTaskNode> ();
      comp_in_task_node->set_stage_node(stage);
      comp_in_task_node->mut_thread_local_id() = thread_local_id;
      comp_in_task_node->SetFwNode();
      comp_in_task_node->SetFwInCopy();
      TaskConnect(comp_in_task_node, NewFinalEdge(), comp_task_node);
      task_nodes_in_stage->comp_in_task_nodes.push_back(comp_in_task_node);
    } else {
      task_nodes_in_stage->comp_in_task_nodes.push_back(comp_task_node);
    }
    // comp_out_task_node
    if (!is_last_stage) {
      CopyHDTaskNode* comp_out_task_node = NewTaskNode<CopyHDTaskNode> ();
      comp_out_task_node->set_stage_node(stage);
      comp_out_task_node->mut_thread_local_id() = thread_local_id;
      comp_out_task_node->SetFwNode();
      comp_out_task_node->SetFwOutCopy();
      TaskConnect(comp_task_node, NewFinalEdge(), comp_out_task_node);
      task_nodes_in_stage->comp_out_task_nodes.push_back(comp_out_task_node);
    } else {
      task_nodes_in_stage->comp_out_task_nodes.push_back(comp_task_node);
    }
  }
  CHECK_EQ(parallel_id, stage->parallel_range().end);
}

void TaskGraph::Stage2HostCompTaskNodes(const StageNode* stage,
                                        TaskNodesInStage* task_nodes_in_stage) {
  HostCompTaskNode* comp_task_node = NewTaskNode<HostCompTaskNode> ();
  comp_task_node->set_stage_node(stage);
  comp_task_node->SetFwNode();
  // since we only support GPU now, it must be a data-op
  int32_t parallel_id = stage->parallel_range().begin;
  comp_task_node->set_parallel_id(parallel_id++);
  CHECK_EQ(parallel_id, stage->parallel_range().end);
  comp_task_node->mut_thread_local_id() = IDManager::Singleton().data_thread_local_id();
  task_nodes_in_stage->comp_in_task_nodes.push_back(comp_task_node);
  task_nodes_in_stage->comp_out_task_nodes.push_back(comp_task_node);
}

void TaskGraph::InitBoxingTaskNodes(Stage2TaskNodesMap* stage2task_nodes) {
  for (const std::unique_ptr<StageNode>& stage : stage_graph_->nodes()) {
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
  InBoxingTaskNode* boxing_task_node = NewTaskNode<InBoxingTaskNode> ();
  boxing_task_node->set_stage_node(stage);
  boxing_task_node->mut_thread_local_id() = IDManager::Singleton().boxing_thread_local_id();
  boxing_task_node->SetFwNode();
  for (TaskNode* comp_in_task_node : task_nodes_in_stage->comp_in_task_nodes) {
    TaskConnect(boxing_task_node, NewFinalEdge(), comp_in_task_node);
  }
  task_nodes_in_stage->in_boxing_task_node = boxing_task_node;
}

void TaskGraph::InitOutBoxingTaskNode(
    const StageNode* stage,
    TaskNodesInStage* task_nodes_in_stage) {
  task_nodes_in_stage->out_boxing_task_node = nullptr;
  if (stage->out_edges().size() == 1
      && task_nodes_in_stage->comp_out_task_nodes.size() == 1) {
    return;
  }
  OutBoxingTaskNode* boxing_task_node = NewTaskNode<OutBoxingTaskNode> ();
  boxing_task_node->set_stage_node(stage);
  boxing_task_node->mut_thread_local_id() = IDManager::Singleton().boxing_thread_local_id();
  boxing_task_node->SetFwNode();
  for (TaskNode* comp_out_task_node : task_nodes_in_stage->comp_out_task_nodes) {
    TaskConnect(comp_out_task_node, NewFinalEdge(), boxing_task_node);
  }
  task_nodes_in_stage->out_boxing_task_node = boxing_task_node;
}

void TaskGraph::ConnectTaskNodes(
    const Stage2TaskNodesMap* stage2task_nodes) {
  for (const std::unique_ptr<StageNode>& cur_stage : stage_graph_->nodes()) {
    const TaskNodesInStage& cur_task_nodes = stage2task_nodes->at(cur_stage.get());
    TaskNode* out_node = cur_task_nodes.out_boxing_task_node;
    if (out_node == nullptr) {
      CHECK_EQ(cur_task_nodes.comp_out_task_nodes.size(), 1);
      out_node = cur_task_nodes.comp_out_task_nodes[0];
    }
    for (const StageEdge* edge : cur_stage->out_edges()) {
      StageNode* succ_stage = edge->dst_node();
      const TaskNodesInStage& succ_task_nodes = stage2task_nodes->at(succ_stage);
      TaskNode* in_node = succ_task_nodes.in_boxing_task_node;
      if (in_node == nullptr) {
        CHECK_EQ(succ_task_nodes.comp_in_task_nodes.size(), 1);
        in_node = succ_task_nodes.comp_in_task_nodes[0];
      }
      if (cur_stage->machine_id() == succ_stage->machine_id()) {
        TaskConnect(out_node, NewFinalEdge(), in_node);
      } else {
        CommNetTaskNode* out_comm_net_node = NewTaskNode<CommNetTaskNode> ();
        CommNetTaskNode* in_comm_net_node = NewTaskNode<CommNetTaskNode> ();
        LOG(FATAL) << "TODO: set node";
        TaskConnect(out_node, NewFinalEdge(), out_comm_net_node);
        TaskConnect(out_comm_net_node, NewFinalEdge(), in_comm_net_node);
        TaskConnect(in_comm_net_node, NewFinalEdge(), in_node);
      }
    }
  }
}

void TaskGraph::BuildBpStruct() {
  std::vector<TaskNode*> turning_node_vec;
  GenerateRelatedBpNodes(&turning_node_vec);
  BackwardConnect(turning_node_vec);
  UpdateSourceAndSink();
}

void TaskGraph::GenerateRelatedBpNodes(
    std::vector<TaskNode*> *turning_node_vec) {
  for (auto task_node = begin(); task_node != end(); ++task_node) {
    if (auto comp_task_node = dynamic_cast<CompTaskNode*> (&(*task_node))) {
      if (comp_task_node->HasOpWithOutDiff()) {
        comp_task_node->BuildAndConnectBpNode();
      } else {
        if (comp_task_node->HasOpWithIndiff()) {
          turning_node_vec->push_back(&(*task_node));
        }
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
    for (TaskEdge* edge : turning_node->in_edges()) {
      TaskNode* bp_pred_node = edge->src_node()->GetBpNode();
      TaskConnect(turning_node, NewFinalEdge(), bp_pred_node);
      bp_node_queue.push(bp_pred_node);
    }
  }
  while (!bp_node_queue.empty()) {
    TaskNode* bp_cur_node = bp_node_queue.front();
    bp_node_queue.pop();
    for (TaskEdge* edge : bp_cur_node->GetFwNode()->in_edges()) {
      TaskNode* bp_pred_node = edge->src_node()->GetBpNode();
      TaskConnect(bp_cur_node, NewFinalEdge(), bp_pred_node);
      bp_node_queue.push(bp_pred_node);
    }
  }
}

} // namespace oneflow
