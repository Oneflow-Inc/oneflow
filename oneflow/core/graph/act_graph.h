#ifndef ONEFLOW_CORE_GRAPH_ACT_GRAPH_H_
#define ONEFLOW_CORE_GRAPH_ACT_GRAPH_H_

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

inline double Duration4ActEvent(const ActEvent& act_event) {
  return act_event.stop_time() - act_event.start_time();
}

class ActNode final : public Node<ActNode, ActEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ActNode);
  explicit ActNode(const ActEvent* act_event, const TaskProto* task_proto)
      : act_event_(act_event), task_proto_(task_proto) {}
  ~ActNode() = default;

  void ForEachProducedRegstDescId(
      const std::function<void(int64_t)>& Handler) const;

  // Getters
  int64_t actor_id() const { return act_event_->actor_id(); }
  int64_t act_id() const { return act_event_->act_id(); }
  double Duration() const { return Duration4ActEvent(*act_event_); }
  const ActEvent& act_event() const { return *act_event_; }
  TaskType task_type() const { return task_proto_->task_type(); }
  std::string VisualStr() const override;

 private:
  const ActEvent* act_event_;
  const TaskProto* task_proto_;
};

class ActGraph final : public Graph<ActNode, ActEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ActGraph);
  explicit ActGraph(const Plan& plan,
                    std::unique_ptr<std::list<ActEvent>>&& act_events);
  ~ActGraph() = default;

  void ForEachRegstDescMeanDuration(
      const std::function<void(int64_t, double)>& Handler) const;
  void ForEachRegstDescIIScale(
      const std::function<void(int64_t, double)>& Handler) const;

  void ToDotFiles(const std::string& dir) const;

  const Plan& plan() const { return *plan_; }

 private:
  void ForEachRegstUidDuration(
      const std::function<void(int64_t regst_desc_id, int64_t act_id,
                               double time)>& Handler) const;
  const Plan* plan_;
  std::unique_ptr<std::list<ActEvent>> act_events_;
  HashMap<std::string, std::list<const ActNode*>> regst_uid2consumer_acts_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_ACT_GRAPH_H_
