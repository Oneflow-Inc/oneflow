#include "graph/task_graph.h"

namespace oneflow {

void TaskGraph::Init(const StageGraph* stage_graph,
                     const IDMap& id_map,
                     bool need_bp) {
  Stage2TaskNodesMap stage2task_nodes;
  InitCompTaskNodes(stage_graph, id_map, &stage2task_nodes);
  InitBoxingTaskNodes(stage_graph, id_map, &stage2task_nodes);
  ConnectTaskNodes(stage_graph, &stage2task_nodes);
  UpdateStartAndStop();
  if (need_bp) {
    BuildBpStruct();
  }
}

void TaskGraph::InitCompTaskNodes(const StageGraph* stage_graph,
                                  const IDMap& id_map,
                                  Stage2TaskNodesMap* stage2task_nodes) {
  for (const std::unique_ptr<Node>& node : stage_graph->node_vec()) {
    auto stage = of_dynamic_cast<const StageNode*> (node.get());
    bool is_first_stage = stage_graph->IsFirstNode(stage);
    bool is_last_stage = stage_graph->IsLastNode(stage);
    if (stage->chain_node()->parallel_desc().engine() == ParallelDesc::Engine::kDevice) {
      Stage2DeviceCompTaskNodes(stage,
                              id_map,
                              &((*stage2task_nodes)[stage]),
                              is_first_stage,
                              is_last_stage);
    } else {
      Stage2HostCompTaskNodes(stage, id_map, &((*stage2task_nodes)[stage]));
    }
  }
}

void TaskGraph::Stage2DeviceCompTaskNodes(
    const StageNode* stage,
    const IDMap& id_map,
    TaskNodesWithinStage* task_nodes_within_stage,
    bool is_first_stage,
    bool is_last_stage) {
  MachineId machine_id = stage->machine_id();
  for (auto device_physical_id : stage->chain_node()->parallel_desc().devices_on_machine(machine_id)) {
    ThreadLocalId thread_local_id =
        id_map.ThreadLocalIdFromDevicePhysicalId(device_physical_id);
    // comp_task_node
    DeviceCompTaskNode* comp_task_node = NewTaskNode<DeviceCompTaskNode> ();
    comp_task_node->set_stage_node(stage);
    comp_task_node->mutable_thread_local_id() = thread_local_id;
    comp_task_node->SetFwNode();
    // comp_in_task_node
    if (!is_first_stage) {
      CopyHDTaskNode* comp_in_task_node = NewTaskNode<CopyHDTaskNode> ();
      comp_in_task_node->set_stage_node(stage);
      comp_in_task_node->mutable_thread_local_id() = thread_local_id;
      comp_in_task_node->SetFwNode();
      comp_in_task_node->SetInCopy();
      Connect(comp_in_task_node, NewTaskEdge(), comp_task_node);
      task_nodes_within_stage->comp_in_task_nodes.push_back(comp_in_task_node);
    } else {
      task_nodes_within_stage->comp_in_task_nodes.push_back(comp_task_node);
    }
    // comp_out_task_node
    if (!is_last_stage) {
      CopyHDTaskNode* comp_out_task_node = NewTaskNode<CopyHDTaskNode> ();
      comp_out_task_node->set_stage_node(stage);
      comp_out_task_node->mutable_thread_local_id() = thread_local_id;
      comp_out_task_node->SetFwNode();
      comp_out_task_node->SetOutCopy();
      Connect(comp_task_node, NewTaskEdge(), comp_out_task_node);
      task_nodes_within_stage->comp_out_task_nodes.push_back(comp_out_task_node);
    } else {
      task_nodes_within_stage->comp_out_task_nodes.push_back(comp_task_node);
    }
  }
}

void TaskGraph::Stage2HostCompTaskNodes(const StageNode* stage,
                                        const IDMap& id_map,
                                        TaskNodesWithinStage* task_nodes_within_stage) {
  HostCompTaskNode* comp_task_node = NewTaskNode<HostCompTaskNode> ();
  comp_task_node->set_stage_node(stage);
  comp_task_node->SetFwNode();
  // since we only support GPU now, it must be a data-op
  comp_task_node->mutable_thread_local_id() = id_map.data_thread_local_id();
  task_nodes_within_stage->comp_in_task_nodes.push_back(comp_task_node);
  task_nodes_within_stage->comp_out_task_nodes.push_back(comp_task_node);
}

void TaskGraph::InitBoxingTaskNodes(const StageGraph* stage_graph,
                                    const IDMap& id_map,
                                    Stage2TaskNodesMap* stage2task_nodes) {
  for (const std::unique_ptr<Node>& node : stage_graph->node_vec()) {
    auto stage = of_dynamic_cast<const StageNode*> (node.get());
    InitInboxingTaskNode(stage, id_map, &(stage2task_nodes->at(stage)));
    InitOutBoxingTaskNode(stage, id_map, &(stage2task_nodes->at(stage)));
  }
}

void TaskGraph::InitInboxingTaskNode(const StageNode* stage,
                                     const IDMap& id_map,
                                     TaskNodesWithinStage* task_nodes_within_stage) {
  task_nodes_within_stage->in_boxing_task_node = nullptr;
  if (stage->in_edges().size() == 1
      && task_nodes_within_stage->comp_in_task_nodes.size() == 1) {
    return;
  }
  BoxingTaskNode* boxing_task_node = NewTaskNode<BoxingTaskNode> ();
  boxing_task_node->set_stage_node(stage);
  boxing_task_node->mutable_thread_local_id() = id_map.boxing_thread_local_id();
  boxing_task_node->SetFwNode();
  boxing_task_node->SetInBoxing();
  for (TaskNode* comp_in_task_node : task_nodes_within_stage->comp_in_task_nodes) {
    Connect(boxing_task_node, NewTaskEdge(), comp_in_task_node);
  }
  task_nodes_within_stage->in_boxing_task_node = boxing_task_node;
}

void TaskGraph::InitOutBoxingTaskNode(
    const StageNode* stage,
    const IDMap& id_map,
    TaskNodesWithinStage* task_nodes_within_stage) {
  task_nodes_within_stage->out_boxing_task_node = nullptr;
  if (stage->out_edges().size() == 1
      && task_nodes_within_stage->comp_out_task_nodes.size() == 1) {
    return;
  }
  BoxingTaskNode* boxing_task_node = NewTaskNode<BoxingTaskNode> ();
  boxing_task_node->set_stage_node(stage);
  boxing_task_node->mutable_thread_local_id() = id_map.boxing_thread_local_id();
  boxing_task_node->SetFwNode();
  boxing_task_node->SetOutBoxing();
  for (TaskNode* comp_out_task_node : task_nodes_within_stage->comp_out_task_nodes) {
    Connect(comp_out_task_node, NewTaskEdge(), boxing_task_node);
  }
  task_nodes_within_stage->out_boxing_task_node = boxing_task_node;
}

void TaskGraph::ConnectTaskNodes(
    const StageGraph* stage_graph,
    const Stage2TaskNodesMap* stage2task_nodes) {
  for (const std::unique_ptr<Node>& node : stage_graph->node_vec()) {
    auto cur_stage = of_dynamic_cast<const StageNode*> (node.get());
    const TaskNodesWithinStage& cur_task_nodes = stage2task_nodes->at(cur_stage);
    TaskNode* out_node = cur_task_nodes.out_boxing_task_node;
    if (out_node == nullptr) {
      CHECK_EQ(cur_task_nodes.comp_out_task_nodes.size(), 1);
      out_node = cur_task_nodes.comp_out_task_nodes[0];
    }
    for (const Edge* edge : cur_stage->out_edges()) {
      auto next_stage = of_dynamic_cast<const StageNode*> (edge->dst_node());
      const TaskNodesWithinStage& next_task_nodes = stage2task_nodes->at(next_stage);
      TaskNode* in_node = next_task_nodes.in_boxing_task_node;
      if (in_node == nullptr) {
        CHECK_EQ(next_task_nodes.comp_in_task_nodes.size(), 1);
        in_node = next_task_nodes.comp_in_task_nodes[0];
      }
      if (cur_stage->machine_id() == next_stage->machine_id()) {
        Connect(out_node, NewTaskEdge(), in_node);
      } else {
        CommNetTaskNode* out_comm_net_node = NewTaskNode<CommNetTaskNode> ();
        CommNetTaskNode* in_comm_net_node = NewTaskNode<CommNetTaskNode> ();
        LOG(FATAL) << "TODO: set node";
        Connect(out_node, NewTaskEdge(), out_comm_net_node);
        Connect(out_comm_net_node, NewTaskEdge(), in_comm_net_node);
        Connect(in_comm_net_node, NewTaskEdge(), in_node);
      }
    }
  }
}

void TaskGraph::GenerateRelatedBpNodes(
    std::function<void(const TaskNode*, TaskNode*)> add_fw_bp_pair,
    const std::unordered_map<const TaskNode*, TaskNode*>& fw_node2bp_node,
    std::vector<TaskNode*> *turning_node_vec) {
  for (auto node_it = begin(); node_it != end(); ++node_it) {
    auto task_node = of_dynamic_cast<TaskNode*> (&(*node_it));
    if (auto comp_task_node = dynamic_cast<CompTaskNode*> (task_node)) {
      if (comp_task_node->HasOpWithOutDiff()) {
        add_fw_bp_pair(task_node, ConstructBpNode(task_node));
      } else {
        if (comp_task_node->HasOpWithIndiff()) {
          turning_node_vec->push_back(task_node);
        }
      }
    } else {
      for (Edge* edge : task_node->in_edges()) {
        if (fw_node2bp_node.find(of_dynamic_cast<TaskNode*> (edge->src_node())) !=
            fw_node2bp_node.end()) {
          add_fw_bp_pair(task_node, ConstructBpNode(task_node));
        }
      }
    }
  }
}

void TaskGraph::BackwardConnect(
    const std::unordered_map<const TaskNode*, TaskNode*>& fw_node2bp_node,
    const std::unordered_map<TaskNode*, const TaskNode*>& bp_node2fw_node,
    const std::vector<TaskNode*>& turning_node_vec) {
  std::queue<TaskNode*> bp_node_queue;
  for (TaskNode* turning_node : turning_node_vec) {
    for (Edge* edge : turning_node->in_edges()) {
      TaskNode* bp_pred_node =
          fw_node2bp_node.at(of_dynamic_cast<TaskNode*>(edge->src_node()));
      Connect(turning_node, NewTaskEdge(), bp_pred_node);
      bp_node_queue.push(bp_pred_node);
    }
  }
  while (!bp_node_queue.empty()) {
    TaskNode* bp_cur_node = bp_node_queue.front();
    bp_node_queue.pop();
    for (Edge* edge : bp_node2fw_node.at(bp_cur_node)->in_edges()) {
      TaskNode* bp_pred_node =
          fw_node2bp_node.at(of_dynamic_cast<TaskNode*>(edge->src_node()));
      Connect(bp_cur_node, NewTaskEdge(), bp_pred_node);
      bp_node_queue.push(bp_pred_node);
    }
  }
}

void TaskGraph::BuildBpStruct() {
  std::unordered_map<const TaskNode*, TaskNode*> fw_node2bp_node;
  std::unordered_map<TaskNode*, const TaskNode*> bp_node2fw_node;
  std::function<void(const TaskNode*, TaskNode*)> add_fw_bp_pair =
      [&fw_node2bp_node, &bp_node2fw_node]
      (const TaskNode* fw_node, TaskNode* bp_node) {
    fw_node2bp_node[fw_node] = bp_node;
    bp_node2fw_node[bp_node] = fw_node;
  };
  std::vector<TaskNode*> turning_node_vec;
  GenerateRelatedBpNodes(add_fw_bp_pair, fw_node2bp_node, &turning_node_vec);
  BackwardConnect(fw_node2bp_node, bp_node2fw_node, turning_node_vec);
  UpdateStartAndStop();
}

} // namespace oneflow
