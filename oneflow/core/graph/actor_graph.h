#ifndef ONEFLOW_CORE_GRAPH_SS_TASK_GRAPH_H_
#define ONEFLOW_CORE_GRAPH_SS_TASK_GRAPH_H_

#include "oneflow/core/graph/graph.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/job/task.pb.h"
#include "oneflow/core/persistence/file_system.h"
//#include "oneflow/core/common/longest_path_visitor.h"
#include "oneflow/core/actor/act_event.pb.h"

namespace oneflow {

class ActorEdge;
class ActorNode final : public Node<ActorNode, ActorEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ActorNode);
  explicit ActorNode(const TaskProto& task_proto) : task_proto_(&task_proto) {}
  ~ActorNode() = default;

  // Getters
  const TaskProto& task_proto() const { return *task_proto_; }
  uint64_t task_id() const { return task_proto().task_id(); }
  TaskType task_type() const { return task_proto().task_type(); }
  uint32_t dfs_order_value() const { return dfs_order_value_; }

  void set_dfs_order_value(uint32_t dfs_order_value) {
    dfs_order_value_ = dfs_order_value;
  }

 private:
  const TaskProto* task_proto_;
  uint32_t dfs_order_value_;
};

class ActorEdge final : public Edge<ActorNode, ActorEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ActorEdge);
  ActorEdge() = default;
  ~ActorEdge() = default;
};

class ActorGraph final : public Graph<ActorNode, ActorEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ActorGraph);
  ActorGraph(const Plan& plan,
             std::unique_ptr<std::list<ActEvent>>&& act_events);
  ~ActorGraph() = default;

  //  Getters
  const Plan& plan() const { return *plan_; }

  const char* TypeName() const { return "ActorGraph"; }
  const ActorNode* GetActorNode(uint64_t task_id) const {
    return task_id2task_.at(task_id);
  }
  //  compute initiation interval
  double InitiationInterval() const;

  void MakeTaskId2AvgDurationHash(
      HashMap<uint64_t, double>* task_id2avg_duration) const;
  void MakeRegstDescId2AvgLifeTimeHash(
      HashMap<uint64_t, double>* regst_desc_id2life_time,
      const std::function<double(uint64_t)>& AvgDuration4TaskId) const;

 private:
  void UpdateDfsOrderValue();
  void DfsForEachNode(const std::function<void(ActorNode*)>& Handler) const;
  const Plan* plan_;
  std::unique_ptr<std::list<ActEvent>> act_events_;
  HashMap<uint64_t, ActorNode*> task_id2task_;
  HashMap<const ActorNode*, std::unordered_set<const ActorNode*>>
      task2ancestors_;
  HashMap<int64_t, std::list<const ActEvent*>> stream_id2act_events_;
};

}  // namespace oneflow
#endif  // ONEFLOW_CORE_GRAPH_SS_TASK_GRAPH_H_
