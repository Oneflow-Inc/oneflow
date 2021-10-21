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
#include "oneflow/core/graph/plan_task_graph.h"

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
        PlanTaskNode* consumer_node = task_id2plan_task_node_.at(consumer_task_id);
        Connect(producer_node, NewEdge(), consumer_node);
      }
    }
  }
}

const TaskProto* PlanTaskGraph::TaskProto4TaskId(int64_t task_id) const {
  return task_id2plan_task_node_.at(task_id)->task_proto();
}

}  // namespace oneflow
