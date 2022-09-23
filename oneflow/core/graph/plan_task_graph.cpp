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
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/graph/plan_task_graph.h"
#include "oneflow/core/graph/task_type.h"

namespace oneflow {

int64_t PlanTaskNode::chain_id() const {
  int64_t chain_id = task_proto_->task_set_info().chain_id();
  CHECK_NE(chain_id, -1);
  return chain_id;
}

PlanTaskGraph::PlanTaskGraph(const Plan& plan) : plan_(&plan) {
  InitNodes();
  InitEdges();
}

void PlanTaskGraph::InitNodes() {
  for (const auto& task : plan_->task()) {
    PlanTaskNode* plan_task_node = new PlanTaskNode(task);
    task_id2plan_task_node_.insert({task.task_id(), plan_task_node});
    AddAllocatedNode(plan_task_node);
  }
}

void PlanTaskGraph::InitEdges() {
  for (const auto& task_id_and_plan_task_node : task_id2plan_task_node_) {
    PlanTaskNode* producer_node = task_id_and_plan_task_node.second;
    for (const auto& pair : producer_node->task_proto()->produced_regst_desc()) {
      for (int64_t consumer_task_id : pair.second.consumer_task_id()) {
        PlanTaskNode* consumer_node = CHECK_JUST(MapAt(task_id2plan_task_node_, consumer_task_id));
        TryConnect(producer_node, consumer_node);
      }
    }
  }
}

void PlanTaskGraph::TryConnect(PlanTaskNode* src, PlanTaskNode* dst) {
  if (edges_.insert({src, dst}).second) { Connect(src, NewEdge(), dst); }
}

const TaskProto* PlanTaskGraph::TaskProto4TaskId(int64_t task_id) const {
  return CHECK_JUST(MapAt(task_id2plan_task_node_, task_id))->task_proto();
}

RankPlanTaskGraph::RankPlanTaskGraph(const Plan& plan,
                                     const HashMap<int64_t, std::string>& comp_task_id2op_name,
                                     const HashSet<PortableCtrlEdge>& portable_ctrl_edges)
    : PlanTaskGraph(plan) {
  InitCtrlEdges(comp_task_id2op_name, portable_ctrl_edges);
}

PlanTaskNode* RankPlanTaskGraph::MutTaskNode4TaskId(int64_t task_id) {
  return CHECK_JUST(MapAt(task_id2plan_task_node_, task_id));
}

void RankPlanTaskGraph::TryConnectByTaskId(int64_t src_task_id, int64_t dst_task_id) {
  TryConnect(MutTaskNode4TaskId(src_task_id), MutTaskNode4TaskId(dst_task_id));
}

void RankPlanTaskGraph::TryConnectBySrcMachineId(
    int64_t src_task_id, const HashMap<int64_t, PlanTaskNode*>& machine_id2task_node) {
  auto* src_task_node = MutTaskNode4TaskId(src_task_id);
  PlanTaskNode* dst_task_node = nullptr;
  if (machine_id2task_node.size() == 1) {
    dst_task_node = machine_id2task_node.begin()->second;
  } else {
    int64_t machine_id = src_task_node->task_proto()->machine_id();
    dst_task_node = CHECK_JUST(MapAt(machine_id2task_node, machine_id));
  }
  TryConnect(src_task_node, dst_task_node);
}

void RankPlanTaskGraph::TryConnectByDstMachineId(
    const HashMap<int64_t, PlanTaskNode*>& machine_id2task_node, int64_t dst_task_id) {
  PlanTaskNode* src_task_node = nullptr;
  auto* dst_task_node = MutTaskNode4TaskId(dst_task_id);
  if (machine_id2task_node.size() == 1) {
    src_task_node = machine_id2task_node.begin()->second;
  } else {
    int64_t machine_id = dst_task_node->task_proto()->machine_id();
    src_task_node = CHECK_JUST(MapAt(machine_id2task_node, machine_id));
  }
  TryConnect(src_task_node, dst_task_node);
}

void RankPlanTaskGraph::TryConnectBetweenCompTaskNodes(
    const HashMap<int64_t, PlanTaskNode*>& machine_id2src_task_node,
    const HashMap<int64_t, PlanTaskNode*>& machine_id2dst_task_node) {
  if (machine_id2src_task_node.size() == 1 && machine_id2dst_task_node.size() >= 1) {
    // there may be some src task nodes missing in the plan.
    // connecting the sole src task node to all dst task nodes will 100% ensure the ctrl-edge order.
    auto* src_task_node = machine_id2src_task_node.begin()->second;
    for (const auto& pair : machine_id2dst_task_node) {
      auto* dst_task_node = pair.second;
      TryConnect(src_task_node, dst_task_node);
    }
  } else if (machine_id2dst_task_node.size() == 1 && machine_id2src_task_node.size() >= 1) {
    // there may be some dst task nodes missing in the plan.
    // connecting all src task nodes to the sole dst task node will 100% ensure the ctrl-edge order.
    auto* dst_task_node = machine_id2dst_task_node.begin()->second;
    for (const auto& pair : machine_id2src_task_node) {
      auto* src_task_node = pair.second;
      TryConnect(src_task_node, dst_task_node);
    }
  } else if (machine_id2src_task_node.size() > 1 && machine_id2dst_task_node.size() > 1) {
    // connect two boxing related comp_task_nodes. this plan have all of them.
    for (const auto& pair : machine_id2src_task_node) {
      int64_t machine_id = pair.first;
      auto* src_task_node = pair.second;
      const auto& iter = machine_id2dst_task_node.find(machine_id);
      if (iter != machine_id2dst_task_node.end()) {
        auto* dst_task_node = iter->second;
        TryConnect(src_task_node, dst_task_node);
      }
    }
  } else {
    UNIMPLEMENTED();
  }
}

void RankPlanTaskGraph::InitCtrlEdges(const HashMap<int64_t, std::string>& comp_task_id2op_name,
                                      const HashSet<PortableCtrlEdge>& portable_ctrl_edges) {
  const auto op_name2machine_id2sole_comp_task_node = [&] {
    HashMap<std::string, HashMap<int64_t, PlanTaskNode*>> map;
    for (const auto& pair : task_id2plan_task_node_) {
      auto* task_node = pair.second;
      if (!IsTransportTaskType::Visit(task_node->task_proto()->task_type())) {
        int64_t task_id = task_node->task_proto()->task_id();
        int64_t machine_id = task_node->task_proto()->machine_id();
        const auto& op_name = CHECK_JUST(MapAt(comp_task_id2op_name, task_id));
        auto* task_node_ptr = &map[op_name][machine_id];
        CHECK(*task_node_ptr == nullptr)
            << "\n----------- old -----------\n"
            << (*task_node_ptr)->task_proto()->DebugString() << "\n----------- new -----------\n"
            << task_node->task_proto()->DebugString();
        *task_node_ptr = task_node;
      }
    }
    return map;
  }();
  for (const auto& edge : portable_ctrl_edges) {
    if (edge.src().has_transport_task_id()) {
      if (edge.dst().has_transport_task_id()) {
        TryConnectByTaskId(edge.src().transport_task_id(), edge.dst().transport_task_id());
      } else if (edge.dst().has_compute_task_op_name()) {
        const auto& machine_id2task_proto = CHECK_JUST(
            MapAt(op_name2machine_id2sole_comp_task_node, edge.dst().compute_task_op_name()));
        TryConnectBySrcMachineId(edge.src().transport_task_id(), machine_id2task_proto);
      } else {
        UNIMPLEMENTED();
      }
    } else if (edge.src().has_compute_task_op_name()) {
      const auto& src_machine_id2task_proto = CHECK_JUST_MSG(
          MapAt(op_name2machine_id2sole_comp_task_node, edge.src().compute_task_op_name()),
          std::stringstream() << "op_names_size: " << op_name2machine_id2sole_comp_task_node.size()
                              << ", new_op_name: " << edge.src().compute_task_op_name());
      if (edge.dst().has_transport_task_id()) {
        TryConnectByDstMachineId(src_machine_id2task_proto, edge.dst().transport_task_id());
      } else if (edge.dst().has_compute_task_op_name()) {
        const auto& dst_machine_id2task_proto = CHECK_JUST(
            MapAt(op_name2machine_id2sole_comp_task_node, edge.dst().compute_task_op_name()));
        TryConnectBetweenCompTaskNodes(src_machine_id2task_proto, dst_machine_id2task_proto);
      } else {
        UNIMPLEMENTED();
      }
    } else {
      UNIMPLEMENTED();
    }
  }
}

}  // namespace oneflow
