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

bool PlanTaskGraph::IsReachable(int64_t src_task_id, int64_t dst_task_id) const {
  return IsReachableToAncestor(task_id2plan_task_node_.at(dst_task_id),
                               task_id2plan_task_node_.at(src_task_id));
}

PlanTaskGraph::PlanTaskGraph(const Plan& plan) : plan_(&plan) {
  InitNodes();
  InitEdges();
  InitNode2Ancestor();
  InitChainId2SortedPlanTaskNode();
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

void PlanTaskGraph::InitNode2Ancestor() {
  TopoForEachNode([&](const PlanTaskNode* node) {
    node->ForEachNodeOnInEdge([&](const PlanTaskNode* prev) {
      node2ancestors_[node].insert(prev);
      node2ancestors_[node].insert(node2ancestors_[prev].begin(), node2ancestors_[prev].end());
    });
  });
}

void PlanTaskGraph::InitChainId2SortedPlanTaskNode() {
  ForEachNode([&](const PlanTaskNode* node) {
    chain_id2sorted_plan_task_nodes_[node->chain_id()].push_back(node);
  });
  for (auto& pair : chain_id2sorted_plan_task_nodes_) {
    std::sort(pair.second.begin(), pair.second.end(),
              [](const PlanTaskNode* lhs, const PlanTaskNode* rhs) {
                return lhs->order_in_graph() < rhs->order_in_graph();
              });
  }
}

bool PlanTaskGraph::IsReachableToAncestor(const PlanTaskNode* node,
                                          const PlanTaskNode* ancestor) const {
  return node2ancestors_.at(node).find(ancestor) != node2ancestors_.at(node).end();
}

const TaskProto* PlanTaskGraph::TaskProto4TaskId(int64_t task_id) const {
  return task_id2plan_task_node_.at(task_id)->task_proto();
}

void PlanTaskGraph::ComputeLifetimeSameChainActorIds(
    const RegstDescProto* regst_desc, HashSet<int64_t>* lifetime_same_chain_actor_ids) const {
  const auto* producer_task_node = task_id2plan_task_node_.at(regst_desc->producer_task_id());
  int64_t chain_id = producer_task_node->chain_id();
  const auto& sorted_plan_task_node = chain_id2sorted_plan_task_nodes_.at(chain_id);
  const PlanTaskNode* last_consumer_task_node = nullptr;
  for (int64_t consumer_task_id : regst_desc->consumer_task_id()) {
    const auto* consumer_task_node = task_id2plan_task_node_.at(consumer_task_id);
    CHECK_EQ(consumer_task_node->chain_id(), chain_id);
    if (last_consumer_task_node == nullptr
        || consumer_task_node->order_in_graph() > last_consumer_task_node->order_in_graph()) {
      last_consumer_task_node = consumer_task_node;
    }
  }
  CHECK_NOTNULL(last_consumer_task_node);
  int64_t start_order_in_graph = producer_task_node->order_in_graph();
  int64_t end_order_in_graph = last_consumer_task_node->order_in_graph();
  auto plan_task_node_it = sorted_plan_task_node.begin();
  for (; plan_task_node_it != sorted_plan_task_node.end()
         && (*plan_task_node_it)->order_in_graph() < start_order_in_graph;
       ++plan_task_node_it) {}
  for (; plan_task_node_it != sorted_plan_task_node.end()
         && (*plan_task_node_it)->order_in_graph() <= end_order_in_graph;
       ++plan_task_node_it) {
    lifetime_same_chain_actor_ids->emplace((*plan_task_node_it)->task_id());
  }
}

}  // namespace oneflow
