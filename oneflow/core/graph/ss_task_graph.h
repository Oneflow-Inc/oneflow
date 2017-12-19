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
  const TaskProto& task_bbproto() const { return *task_proto_; }
  uint64_t task_id() const { return task_proto().id(); }
  TaskType type() const { return task_proto().type(); }

  virtual std::string VisualStr() const;
  std::string GetDeviceName() const;
  std::string GlobalUniqueStreamName(uint64_t stream_id) const;

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
  SSTaskGraph(const Plan& plan, const std::list<ActEvent>& act_events);
  ~SSTaskGraph() = default;

  //  Getters
  const Plan& plan() const { return *plan_; }
  const UtilizationPackageProto& utilization_package() const {
    return utilization_package_;
  }

  const char* TypeName() const { return "SSTaskGraph"; }
  const SSTaskNode* GetSSTaskNode(uint64_t task_id) const {
    return task_id2task_.at(task_id);
  }
  //  compute initiation interval
  float InitiationInterval() const;

  void MakeTaskId2AvgDurationHash(
      std::unordered_map<uint64_t, float>* task_id2avg_duration) const;
  void MakeRegstDescId2AvgLifeTimeHash(
      std::unordered_map<uint64_t, float>* regst_desc_id2life_time,
      const std::function<float(uint64_t)>& AvgDuration4TaskId) const;

 private:
  void InitGraph();
  void InitUtilization(const std::list<ActEvent>& act_events);
  void InitUtilizationIndexes();
  void UpdateAncestors();
  bool IsAncestor(const SSTaskNode* asc, const SSTaskNode* node) const;
  void RemoveMeaninglessEdges();
  bool ReachableWithoutEdge(const SSTaskEdge* edge) const;
  void ForEachRegstDesc(
      const std::function<void(const RegstDescProto&)>& DoEach) const;
  float AvgLifeTime(const LongestPathVisitor<const SSTaskNode*>& lpath_visitor,
                    const SSTaskNode* start_task, const SSTaskNode* end_task,
                    const std::function<double(const SSTaskNode* task)>&
                        AvgDuration4Task) const;

  const Plan* plan_;
  std::unordered_map<uint64_t, SSTaskNode*> task_id2task_;
  std::unordered_map<const SSTaskNode*, std::unordered_set<const SSTaskNode*>>
      task2ancestors_;
  std::unordered_map<uint64_t, std::list<const UtilizationProto*>>
      task_id2utilization_protos_;
  std::unordered_map<std::string, std::list<const UtilizationProto*>>
      stream2utilization_protos_;
  std::unordered_map<uint64_t, std::unordered_set<std::string>>
      task_id2streams_;
  UtilizationPackageProto utilization_package_;
  std::list<ActEvent> comp_act_events_;
};

}  // namespace oneflow
#endif  // ONEFLOW_CORE_GRAPH_SS_TASK_GRAPH_H_
