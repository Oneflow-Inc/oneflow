#ifndef ONEFLOW_CORE_GRAPH_CHAIN_ACT_GRAPH_H_
#define ONEFLOW_CORE_GRAPH_CHAIN_ACT_GRAPH_H_

#include "oneflow/core/graph/graph.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/job/task.pb.h"
#include "oneflow/core/actor/act_event.pb.h"

namespace oneflow {

class ChainActNode;

struct RegstAct {
  RegstAct(int64_t input_regst_desc_id, const ActEvent* input_producer_act_event,
           std::list<const ActEvent*> input_consumer_act_events)
      : regst_desc_id(input_regst_desc_id),
        producer_act_event(input_producer_act_event),
        consumer_act_events(input_consumer_act_events) {}

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

inline double Duration4ActEvent(const ActEvent& act_event) {
  return act_event.stop_time() - act_event.start_time();
}

inline double Duration4RegstActConsumerPath(std::shared_ptr<RegstActCtx> regst_act_ctx,
                                            const ActEvent* consumer_act_event,
                                            const ChainActNode* consumer_node) {
  return regst_act_ctx->node2duration_to_producer.at(consumer_node)
         + consumer_act_event->stop_time()
         - regst_act_ctx->regst_act->producer_act_event->start_time();
}

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
  ChainActNode(std::list<std::unique_ptr<ActEvent>>&& act_events);
  ~ChainActNode() = default;

  // ForEach
  void ForEachActEvent(const std::function<void(const ActEvent*)>& Handler) const;
  void ForEachProducedRegstAct(const std::function<void(const RegstAct*)>& Handler) const;
  void ForEachLastConsumedRegstAct(const std::function<void(const RegstAct*)>& Handler) const;

  // Getters
  int64_t act_id() const { return act_events_.front()->act_id(); }

  // Setters
  void AddProducedRegstActs(std::unique_ptr<RegstAct>&& regst_act) {
    produced_regst_acts_.push_back(std::move(regst_act));
  }
  void AddLastConsumedRegstActs(const RegstAct* regst_act) {
    last_consumed_regst_acts_.push_back(regst_act);
  }

 private:
  std::list<std::unique_ptr<ActEvent>> act_events_;
  std::list<std::unique_ptr<RegstAct>> produced_regst_acts_;
  std::list<const RegstAct*> last_consumed_regst_acts_;
};

class ChainActGraph final : public Graph<const ChainActNode, const ChainActEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ChainActGraph);
  ChainActGraph(const Plan& plan, std::list<std::unique_ptr<ActEvent>>&& act_events);
  ~ChainActGraph() = default;

  // ForEach
  void ForEachRegstDescConsumerPathMeanDuration(
      const std::function<void(int64_t, int64_t, double)>& Handler) const;
  void ForEachRegstDescConsumerPathIIScale(
      const std::function<void(int64_t, int64_t, double)>& Handler) const;
  void ForEachActEvent(const std::function<void(const ActEvent*)>& Handler) const;

  // Getters
  const TaskProto& GetTaskProto(int64_t actor_id) const {
    return *task_id2task_proto_.at(actor_id);
  }
  bool IsActEventWithConsumer(const ActEvent* act_event) const {
    return act_event_with_consumer_.find(act_event) != act_event_with_consumer_.end();
  }

 private:
  const ChainActNode* Node4ActEvent(const ActEvent* act_event) const;
  std::function<const int64_t&(const ChainActNode*)> MakeGetterTopoOrderValue4Node() const;

  void InitNodes(
      std::list<std::unique_ptr<ActEvent>>&& act_events,
      HashMap<std::pair<int64_t, int64_t>, const ActEvent*>* regst_uid2producer_act_event);
  void InitEdges(
      const HashMap<std::pair<int64_t, int64_t>, const ActEvent*>& regst_uid2producer_act_event,
      HashMap<std::pair<int64_t, int64_t>, std::list<const ActEvent*>>*
          regst_uid2consumer_act_events);
  void InitNodeProducedRegstActs(
      const HashMap<std::pair<int64_t, int64_t>, const ActEvent*>& regst_uid2producer_act_event,
      const HashMap<std::pair<int64_t, int64_t>, std::list<const ActEvent*>>&
          regst_uid2consumer_act_events);
  void InitTaskId2TaskProto();
  void InitNodeLastConsumedRegstActs();
  void ForEachInEdge(const ChainActNode* node,
                     const std::function<void(const ChainActEdge*)>& Handler) const;
  void ForEachOutEdge(const ChainActNode* node,
                      const std::function<void(const ChainActEdge*)>& Handler) const;
  void TopoForEachChainActNode(const std::function<void(const ChainActNode*)>& Handler) const;
  void ForEachRegstActConsumerPathDuration(
      const std::function<void(int64_t, int64_t, double)>& Handler) const;
  void CalcRegstActNodePathDuration(std::shared_ptr<RegstActCtx> regst_act_ctx,
                                    const ChainActNode* node) const;

  const Plan* plan_;
  HashMap<int64_t, const TaskProto*> task_id2task_proto_;
  HashSet<const ActEvent*> act_event_with_consumer_;
  HashMap<const ActEvent*, ChainActNode*> act_event2chain_node_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_CHAIN_ACT_GRAPH_H_
