#ifndef ONEFLOW_CORE_GRAPH_SS_TASK_GRAPH_H_
#define ONEFLOW_CORE_GRAPH_SS_TASK_GRAPH_H_

#include "oneflow/core/graph/graph.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/job/task.pb.h"
#include "oneflow/core/persistence/file_system.h"
#include "oneflow/core/actor/act_event.pb.h"

namespace oneflow {

class ActNode;
class ActEdge final : public Edge<ActNode, ActEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ActEdge);
  ActEdge() = default;
  ~ActEdge() = default;
};

class ActNode final : public Node<ActNode, ActEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ActNode);
  explicit ActNode(const ActEvent& act_event) : act_event_(act_event) {}
  ~ActNode() = default;

  // Getters
  int64_t actor_id() const { return act_event_.actor_id(); }

 private:
  ActEvent act_event_;
};

class ActGraph final : public Graph<ActNode, ActEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ActGraph);
  explicit ActGraph(const Plan& plan,
                    std::unique_ptr<std::list<ActEvent>>&& act_events);
  ~ActGraph() = default;

  void ForEachRegstLifeTime(
      const std::function<void(int64_t, double)>& Handler) const;

  const Plan& plan() const { return *plan_; }

 private:
  double CalcLongestPathTime(const ActNode* start_node,
                             const std::list<const ActNode*>& end_nodes) const;
  void CreateNodes();
  void ConnectNodes();
  const Plan* plan_;
  std::unique_ptr<std::list<ActEvent>> act_events_;
  HashMap<int64_t, std::list<ActNode*>> actor_id2act_nodes_;
  HashMap<std::string, ActNode*> act_uid2act_node_;
};

}  // namespace oneflow
#endif  // ONEFLOW_CORE_GRAPH_SS_TASK_GRAPH_H_
