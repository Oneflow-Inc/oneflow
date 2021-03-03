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
#ifndef ONEFLOW_CORE_GRAPH_PLAN_TASK_GRAPH_H_
#define ONEFLOW_CORE_GRAPH_PLAN_TASK_GRAPH_H_

#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/graph/graph.h"

namespace oneflow {

class PlanTaskNode;

class PlanTaskEdge final : public Edge<PlanTaskNode, PlanTaskEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PlanTaskEdge);
  PlanTaskEdge() = default;
  ~PlanTaskEdge() = default;
};

class PlanTaskNode final : public Node<PlanTaskNode, PlanTaskEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PlanTaskNode);
  explicit PlanTaskNode(const TaskProto& task_proto) : task_proto_(&task_proto) {}
  ~PlanTaskNode() = default;

  const TaskProto* task_proto() const { return task_proto_; }
  int64_t task_id() const { return task_proto_->task_id(); }
  int64_t chain_id() const;
  int64_t order_in_graph() const { return task_proto_->task_set_info().order_in_graph(); }

 private:
  const TaskProto* task_proto_;
};

class PlanTaskGraph final : public Graph<const PlanTaskNode, PlanTaskEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PlanTaskGraph);
  explicit PlanTaskGraph(const Plan& plan);
  ~PlanTaskGraph() = default;

  void ComputeLifetimeSameChainActorIds(const RegstDescProto* regst_desc,
                                        HashSet<int64_t>* lifetime_same_chain_actor_ids) const;
  bool IsReachable(int64_t src_task_id, int64_t dst_task_id) const;

  const TaskProto* TaskProto4TaskId(int64_t task_id) const;
  const Plan& plan() const { return *plan_; }

 private:
  void InitNodes();
  void InitEdges();
  void InitNode2Ancestor();
  void InitChainId2SortedPlanTaskNode();
  bool IsReachableToAncestor(const PlanTaskNode* node, const PlanTaskNode* ancestor) const;

  const Plan* plan_;
  HashMap<int64_t, PlanTaskNode*> task_id2plan_task_node_;
  HashMap<const PlanTaskNode*, HashSet<const PlanTaskNode*>> node2ancestors_;
  HashMap<int64_t, std::vector<const PlanTaskNode*>> chain_id2sorted_plan_task_nodes_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_PLAN_TASK_GRAPH_H_
