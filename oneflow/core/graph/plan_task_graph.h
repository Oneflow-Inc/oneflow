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
  int64_t area_id() const { return task_proto_->task_set_info().area_id(); }
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
  void AssertThereIsOnlyOneTopoOrder(const HashSet<int64_t>& same_stream_nodes) const;

  void SortByProducerTaskOrderInGraph(
      const std::list<const RegstDescProto*>& regst_descs,
      const std::function<void(const RegstDescProto*)>& Handler) const;
  const TaskProto* TaskProto4TaskId(int64_t task_id) const {
    return task_id2plan_task_node_.at(task_id)->task_proto();
  }
  bool IsReachableInSameArea(int64_t src_task_id, int64_t dst_task_id) const;

 private:
  void InitNodes(const Plan& plan);
  void InitEdges();
  void InitNode2Ancestor();
  bool IsAnyNodeReachableToAncestor(const HashSet<const PlanTaskNode*>& nodes,
                                    const PlanTaskNode* ancestor) const;
  bool IsReachableToAncestor(const PlanTaskNode* node, const PlanTaskNode* ancestor) const;
  void ComputeLifetimeActorIds(const RegstDescProto* regst_desc,
                               HashSet<int64_t>* lifetime_actor_ids,
                               const std::function<bool(int64_t)>& IsAllowed) const;

  HashMap<int64_t, PlanTaskNode*> task_id2plan_task_node_;
  HashMap<const PlanTaskNode*, HashSet<const PlanTaskNode*>> node2ancestors_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_PLAN_TASK_GRAPH_H_
