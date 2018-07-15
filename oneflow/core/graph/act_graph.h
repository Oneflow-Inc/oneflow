#ifndef ONEFLOW_CORE_GRAPH_ACT_GRAPH_H_
#define ONEFLOW_CORE_GRAPH_ACT_GRAPH_H_

#include "oneflow/core/graph/graph.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/job/task.pb.h"
#include "oneflow/core/persistence/file_system.h"
#include "oneflow/core/actor/act_event.pb.h"
#include "oneflow/core/common/range.h"

namespace oneflow {

inline double Duration4ActEvent(const ActEvent& act_event) {
  return act_event.stop_time() - act_event.start_time();
}

class ChainActNode;
class ChainActEdge final : public Edge<ChainActNode, ChainActEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ChainActEdge);
  ChainActEdge() = default;
  ~ChainActEdge() = default;

  // Getters
  double duration() const { return duration_; }
  // Setters
  void set_duration(double duration) { duration_ = duration; }

 private:
  double duration_;
};

class ChainActNode final : public Node<ChainActNode, ChainActEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ChainActNode);
  ChainActNode(std::list<ActEvent*> act_events);
  ~ChainActNode() = default;

  // Getters
  std::list<const ActEvent*> act_events() const { return act_events_; }
  int64_t act_id() const { return act_events_.front()->act_id(); }
  int64_t depth() const { return depth_; }
  int64_t topo_id() const { return topo_id_; }
  // Setters
  void set_depth(int64_t depth) { depth_ = depth; }
  void set_topo_id(int64_t topo_id) { topo_id_ = topo_id; }

 private:
  std::list<const ActEvent*> act_events_;
  int64_t depth_;
  int64_t topo_id_;
};

class ChainActGraph final : public Graph<ChainActNode, ChainActEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ChainActGraph);
  ChainActGraph(const Plan& plan, std::unique_ptr<std::list<ActEvent>>&& act_events);
  ~ChainActGraph() = default;

  // ForEach
  void ForEachRegstDescConsumerPathMeanDuration(
      const std::function<void(int64_t, int64_t, double)>& Handler) const;
  void ForEachRegstDescConsumerPathIIScale(
      const std::function<void(int64_t, int64_t, double)>& Handler) const;
  void ForEachActEvent(const std::function<void(const ActEvent*)>& Handler) const;

  // Getters
  bool ActEventHasConsumer(const ActEvent* act_event) const {
    return act_event2has_consumer_.at(act_event);
  }
  const std::list<const ChainActNode*>& Nodes4Depth(int64_t depth) const {
    return depth2nodes_.at(depth);
  }
  ChainActNode* ProducerNode4RegstUid(const std::string& regst_uid) const {
    return act_event2chain_node_.at(regst_uid2producer_act_event_.at(regst_uid));
  }
  const std::list<const ActEvent*>& ConsumerActEvents4RegstUid(const std::string& regst_uid) const {
    return regst_uid2consumer_act_events_.at(regst_uid);
  }
  const ActEvent* ProducerActEvent4RegstUid(const std::string& regst_uid) const {
    return regst_uid2producer_act_event_.at(regst_uid);
  }
  const std::list<const ChainActNode*>& AllConsumers4Producer(const ChainActNode* producer) const {
    return producer2consumers_.at(producer);
  }
  const ChainActNode* Node4ActEvent(const ActEvent* act_event) const {
    return act_event2chain_node_.at(act_event);
  }
  const TaskProto& GetTaskProto(int64_t actor_id) const {
    return *task_id2task_proto_.at(actor_id);
  }

 private:
  void InitNodes();
  void InitEdges();
  void InitDepth7TopoId();
  void InitTaskId2TaskProto();
  void ForEachInEdge(const ChainActNode* node,
                     const std::function<void(const ChainActEdge*)>& Handler) const;
  void TopoForEachChainActNode(std::list<ChainActNode*>& starts,
                               const std::function<void(ChainActNode*)>& Handler) const;
  void TopoForEachChainActNode2(std::list<ChainActNode*>& starts,
                                const std::function<void(ChainActNode*)>& Handler) const;
  void ForEachDepthRangeRegstUids(
      const std::function<void(const Range& range, const std::list<std::string>& regst_uids)>&
          Handler) const;
  void ForEachDepthRangeRegstUids2(
      const std::function<void(const Range& range, const std::list<std::string>& regst_uids)>&
          Handler) const;
  void ForEachRegstUidConsumerPathDuration(
      const std::function<void(const std::string&, int64_t, double)>& Handler) const;
  void ForEachRegstUidConsumerPathDuration() const;
  const Plan* plan_;
  std::unique_ptr<std::list<ActEvent>> act_events_;
  HashMap<int64_t, std::list<const ChainActNode*>> depth2nodes_;
  HashMap<int64_t, const TaskProto*> task_id2task_proto_;
  HashMap<const ActEvent*, bool> act_event2has_consumer_;
  HashMap<std::string, const ActEvent*> regst_uid2producer_act_event_;
  HashMap<std::string, std::list<const ActEvent*>> regst_uid2consumer_act_events_;
  HashMap<const ActEvent*, ChainActNode*> act_event2chain_node_;
  HashMap<const ChainActNode*, std::list<const ChainActNode*>> producer2consumers_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_ACT_GRAPH_H_
