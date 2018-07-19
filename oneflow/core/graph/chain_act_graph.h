#ifndef ONEFLOW_CORE_GRAPH_CHAIN_ACT_GRAPH_H_
#define ONEFLOW_CORE_GRAPH_CHAIN_ACT_GRAPH_H_

#include "oneflow/core/graph/graph.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/job/task.pb.h"
#include "oneflow/core/actor/act_event.pb.h"

namespace oneflow {

inline double Duration4ActEvent(const ActEvent& act_event) {
  return act_event.stop_time() - act_event.start_time();
}

class ChainActNode;

struct RegstAct {
  RegstAct(int64_t in_regst_desc_id, const ActEvent* in_producer_act_event,
           std::list<const ActEvent*> in_consumer_act_events)
      : regst_desc_id(in_regst_desc_id),
        producer_act_event(in_producer_act_event),
        consumer_act_events(in_consumer_act_events) {}

  int64_t regst_desc_id;
  const ActEvent* producer_act_event;
  std::list<const ActEvent*> consumer_act_events;
  HashSet<const ChainActNode*> actual_producer_outs;
};

struct RegstActCtx {
  RegstActCtx(const RegstAct* input_regst_act, const ChainActNode* producer)
      : regst_act(input_regst_act) {
    node2duration_to_producer[producer] = 0;
  }

  const RegstAct* regst_act;
  HashMap<const ChainActNode*, double> node2duration_to_producer;
};

class ChainActEdge final : public Edge<ChainActNode, ChainActEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ChainActEdge);
  ChainActEdge(double duration) : duration_(duration) {}
  ~ChainActEdge() = default;

  // Getters
  const double duration() const { return duration_; }

 private:
  const double duration_;
};

class ChainActNode final : public Node<ChainActNode, ChainActEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ChainActNode);
  ChainActNode(std::list<ActEvent*> act_events);
  ~ChainActNode() = default;

  // Getters
  int64_t act_id() const { return act_events_.front()->act_id(); }
  std::list<const ActEvent*> act_events() const { return act_events_; }
  std::list<const RegstAct*> ProducedRegstActs() const { return produced_regst_acts_; }
  std::list<const RegstAct*> LastConsumedRegstActs() const { return last_consumed_regst_acts_; }

  // Setters
  void AddProducedRegstActs(RegstAct* regst_act) { produced_regst_acts_.push_back(regst_act); }
  void AddLastConsumedRegstActs(RegstAct* regst_act) {
    last_consumed_regst_acts_.push_back(regst_act);
  }

 private:
  std::list<const ActEvent*> act_events_;
  std::list<const RegstAct*> produced_regst_acts_;
  std::list<const RegstAct*> last_consumed_regst_acts_;
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
  bool IsActEventWithConsumer(const ActEvent* act_event) const {
    return act_event_with_consumer_.find(act_event) != act_event_with_consumer_.end();
  }
  const TaskProto& GetTaskProto(int64_t actor_id) const {
    return *task_id2task_proto_.at(actor_id);
  }

 private:
  const ChainActNode* Node4ActEvent(const ActEvent* act_event) const {
    return act_event2chain_node_.at(act_event);
  }

  void InitNodes(
      HashMap<std::pair<int64_t, int64_t>, const ActEvent*>* regst_uid2producer_act_event);
  void InitEdges(
      const HashMap<std::pair<int64_t, int64_t>, const ActEvent*>& regst_uid2producer_act_event,
      HashMap<std::pair<int64_t, int64_t>, std::list<const ActEvent*>>*
          regst_uid2consumer_act_events);
  void InitRegstActs7NodeProducedAndLastConsumedRegstActs(
      const HashMap<std::pair<int64_t, int64_t>, const ActEvent*>& regst_uid2producer_act_event,
      const HashMap<std::pair<int64_t, int64_t>, std::list<const ActEvent*>>&
          regst_uid2consumer_act_events);
  void InitTaskId2TaskProto();
  void ForEachInEdge(const ChainActNode* node,
                     const std::function<void(const ChainActEdge*)>& Handler) const;
  void ForEachOutEdge(const ChainActNode* node,
                      const std::function<void(const ChainActEdge*)>& Handler) const;
  void TopoForEachChainActNode(const std::function<void(ChainActNode*)>& Handler) const;
  void ForEachRegstActConsumerPathDuration(
      const std::function<void(int64_t, int64_t, double)>& Handler) const;
  void CalcRegstActNodePathDuration(std::shared_ptr<RegstActCtx> regst_act_ctx,
                                    const ChainActNode* node) const;
  void ForEachConsumerPathDuration(
      std::shared_ptr<RegstActCtx> regst_act_ctx,
      const std::function<void(int64_t, int64_t, double)>& Handler) const;

  const Plan* plan_;
  std::unique_ptr<std::list<ActEvent>> act_events_;
  HashMap<int64_t, const TaskProto*> task_id2task_proto_;
  HashSet<const ActEvent*> act_event_with_consumer_;
  HashMap<const ActEvent*, ChainActNode*> act_event2chain_node_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_CHAIN_ACT_GRAPH_H_
