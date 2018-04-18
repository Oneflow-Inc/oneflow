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
  void set_depth(int64_t depth) { depth_ = depth; }

 private:
  const ActEvent* act_event_;
  const TaskProto* task_proto_;
  int64_t depth_;
  HashMap<std::string, std::list<const ActNode*>> regst_uid2consumer_nodes_;
};

class DepthRangeActSubGraph;
class RegstActSubGraph;

class ActGraph final : public Graph<ActNode, ActEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ActGraph);
  ActGraph(const Plan& plan, std::unique_ptr<std::list<ActEvent>>&& act_events);
  ~ActGraph() = default;

  void ForEachRegstDescConsumerPathMeanDuration(
      const std::function<void(int64_t, int64_t, double)>& Handler) const;
  void ForEachRegstDescConsumerPathIIScale(
      const std::function<void(int64_t, int64_t, double)>& Handler) const;
  void ToDotFiles(const std::string& dir) const;

  // Getters
  const Plan& plan() const { return *plan_; }
  const TaskProto& GetTaskProto(int64_t actor_id) const {
    return *task_id2task_proto_.at(actor_id);
  }
  const HashMap<int64_t, int64_t>& actor_id2act_cnt() const {
    return actor_id2act_cnt_;
  }
  const HashMap<int64_t, double>& actor_id2total_act_time() const {
    return actor_id2total_act_time_;
  }
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

 private:
  void ForEachRegstUidConsumerPathDuration(
      const std::function<void(const std::string&, int64_t, double)>& Handler)
      const;
  void InitNodes();
  void InitEdges();
  void InitDepth();
  void InitTaskId2TaskProto();
  void InitActorStatistics();
  void ForEachDepthRangeRegstUids(
      const std::function<void(const Range& range,
                               const std::list<std::string>& regst_uids)>&
          Handler) const;
  void ForEachDepthRangeSubActGraph(
      const std::function<void(const DepthRangeActSubGraph&)>& Handler) const;
  void ForEachRegstActSubGraph(
      const std::function<void(const RegstActSubGraph&)>& Handler) const;
  void TopoForEachActNode(const std::list<ActNode*>& starts,
                          const std::function<void(ActNode*)>& Handler) const;

  const Plan* plan_;
  std::unique_ptr<std::list<ActEvent>> act_events_;
  HashMap<int64_t, const TaskProto*> task_id2task_proto_;
  HashMap<std::string, ActNode*> regst_uid2producer_node_;
  HashMap<std::string, std::list<const ActNode*>> regst_uid2consumer_nodes_;
  HashMap<int64_t, std::list<const ActNode*>> depth2nodes_;
  HashMap<int64_t, int64_t> actor_id2act_cnt_;
  HashMap<int64_t, double> actor_id2total_act_time_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_ACT_GRAPH_H_
