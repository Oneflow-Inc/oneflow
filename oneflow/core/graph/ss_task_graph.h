#ifndef ONEFLOW_CORE_GRAPH_SS_TASK_GRAPH_H_
#define ONEFLOW_CORE_GRAPH_SS_TASK_GRAPH_H_

#include "oneflow/core/graph/graph.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/job/task.pb.h"
#include "oneflow/core/persistence/file_system.h"
#include "oneflow/core/common/longest_path_visitor.h"
#include "oneflow/core/actor/act_event.pb.h"

namespace oneflow {

class SSTaskEdge;
class SSTaskNode final : public Node<SSTaskNode, SSTaskEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SSTaskNode);
  explicit SSTaskNode(const TaskProto& task_proto) : task_proto_(&task_proto) {}
  ~SSTaskNode() = default;

  // Getters
  const TaskProto& task_proto() const { return *task_proto_; }
  uint64_t task_id() const { return task_proto().task_id(); }
  TaskType type() const { return task_proto().task_type(); }

  // virtual std::string VisualStr() const;
  // std::string GetDeviceName() const;

 private:
  const TaskProto* task_proto_;
};

class SSTaskEdge final : public Edge<SSTaskNode, SSTaskEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SSTaskEdge);
  SSTaskEdge() = default;
  ~SSTaskEdge() = default;
};

//  static schedule task graph
class SSTaskGraph final : public Graph<SSTaskNode, SSTaskEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SSTaskGraph);
  SSTaskGraph(const Plan& plan,
              std::unique_ptr<std::list<ActEvent>>&& act_events);
  ~SSTaskGraph() = default;

  //  Getters
  const Plan& plan() const { return *plan_; }

  const char* TypeName() const { return "SSTaskGraph"; }
  const SSTaskNode* GetSSTaskNode(uint64_t task_id) const {
    return task_id2task_.at(task_id);
  }
  //  compute initiation interval
  double InitiationInterval() const;

  void MakeTaskId2AvgDurationHash(
      std::unordered_map<uint64_t, double>* task_id2avg_duration) const;
  void MakeRegstDescId2AvgLifeTimeHash(
      std::unordered_map<uint64_t, double>* regst_desc_id2life_time,
      const std::function<double(uint64_t)>& AvgDuration4TaskId) const;

 private:
  /*
  void InitActEvents();
  void InitGraph();
  void UpdateAncestors();
  bool IsAncestor(const SSTaskNode* asc, const SSTaskNode* node) const;
  void RemoveMeaninglessEdges();
  bool ReachableWithoutEdge(const SSTaskEdge* edge) const;
  void ForEachRegstDesc(
      const std::function<void(const RegstDescProto&)>& DoEach) const;
  double AvgLifeTime(const LongestPathVisitor<const SSTaskNode*>& lpath_visitor,
                     const SSTaskNode* start_task, const SSTaskNode* end_task,
                     const std::function<double(const SSTaskNode* task)>&
                         AvgDuration4Task) const;
  */

  const Plan* plan_;
  std::unique_ptr<std::list<ActEvent>> act_events_;
  std::unordered_map<uint64_t, SSTaskNode*> task_id2task_;
  std::unordered_map<const SSTaskNode*, std::unordered_set<const SSTaskNode*>>
      task2ancestors_;
  std::unordered_map<int64_t, std::list<const ActEvent*>> task_id2act_events_;
  std::unordered_map<int64_t, std::list<const ActEvent*>> stream_id2act_events_;
};

}  // namespace oneflow
#endif  // ONEFLOW_CORE_GRAPH_SS_TASK_GRAPH_H_
