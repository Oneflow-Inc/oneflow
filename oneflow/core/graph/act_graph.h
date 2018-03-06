#ifndef ONEFLOW_CORE_GRAPH_ACT_GRAPH_H_
#define ONEFLOW_CORE_GRAPH_ACT_GRAPH_H_

#include "oneflow/core/graph/graph.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/job/task.pb.h"
#include "oneflow/core/persistence/file_system.h"
#include "oneflow/core/actor/act_event.pb.h"
#include "oneflow/core/common/range.h"

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
  ActNode(const ActEvent* act_event, const TaskProto* task_proto)
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
  int64_t depth() const { return depth_; }

  const std::list<const ActNode*>& ConsumerNodes4RegstInfo(
      const std::string& regst_uid) const {
    return regst_uid2consumer_nodes_.at(regst_uid);
  }

  // Setters
  void AddConsumerNode(const std::string& regst_uid,
                       const ActNode* consumer_node);

 private:
  friend class ActGraph;
  void set_depth(int64_t depth) const { depth_ = depth; }

  const ActEvent* act_event_;
  const TaskProto* task_proto_;
  mutable int64_t depth_;
  HashMap<std::string, std::list<const ActNode*>> regst_uid2consumer_nodes_;
};

class RegstActSubGraph;

class ActGraph final : public Graph<ActNode, ActEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ActGraph);
  ActGraph(const Plan& plan,
                    std::unique_ptr<std::list<ActEvent>>&& act_events);
  ~ActGraph() = default;

  void ForEachRegstDescMeanDuration(
      const std::function<void(int64_t, double)>& Handler) const;
  void ForEachRegstDescIIScale(
      const std::function<void(int64_t, double)>& Handler) const;

  void ToDotFiles(const std::string& dir) const;

  const Plan& plan() const { return *plan_; }

 private:
  friend class DepthRangeActSubGraph;
  void ForEachRegstUidDuration(
      const std::function<void(const std::string&, double)>& Handler) const;

  void InitNodes();
  void InitEdges();
  void InitDepth();

  const std::list<const ActNode*>& Nodes4Depth(int64_t depth) const {
    return depth2nodes_.at(depth);
  }
  const ActNode* ProducerNode4RegstUid(const std::string& regst_uid) const {
    return regst_uid2producer_node_.at(regst_uid);
  }

  const std::list<const ActNode*>& ConsumerNodes4RegstUid(
      const std::string& regst_uid) const {
    return regst_uid2consumer_nodes_.at(regst_uid);
  }
  void ForEachDepthRangeRegstUids(
      const std::function<void(const Range& range,
                               const std::list<std::string>& regst_uids)>&
          Handler) const;
  void ForEachRegstActSubGraph(
      const std::function<void(const RegstActSubGraph&)>& Handler) const;
  void TopoForEachActNode(
      const std::list<const ActNode*>& starts,
      const std::function<void(const ActNode*)>& Handler) const;

  const Plan* plan_;
  std::unique_ptr<std::list<ActEvent>> act_events_;
  HashMap<std::string, ActNode*> regst_uid2producer_node_;
  HashMap<std::string, std::list<const ActNode*>> regst_uid2consumer_nodes_;
  HashMap<int64_t, std::list<const ActNode*>> depth2nodes_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_ACT_GRAPH_H_
