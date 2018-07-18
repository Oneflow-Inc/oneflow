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
  RegstAct(int64_t input_regst_desc_id, const ActEvent* input_producer_act_event)
      : regst_desc_id(input_regst_desc_id), producer_act_event(input_producer_act_event) {}

  int64_t regst_desc_id;
  const ActEvent* producer_act_event;
  std::list<const ActEvent*> consumer_act_events;
  HashSet<const ChainActNode*> actual_producer_outs;
};

struct RegstActCtx {
  RegstActCtx(const RegstAct* input_regst_act) : regst_act(input_regst_act) {}

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
  int64_t topo_order_value() const { return topo_order_value_; }
  std::list<const ActEvent*> act_events() const { return act_events_; }
  std::list<const RegstAct*> produced_regst_acts() const { return produced_regst_acts_; }
  std::list<const RegstAct*> last_consumed_regst_acts() const { return last_consumed_regst_acts_; }

  // Setters
  void set_topo_order_value(int64_t topo_order_value) { topo_order_value_ = topo_order_value; }
  void set_produced_regst_acts(RegstAct* regst_act) { produced_regst_acts_.push_back(regst_act); }
  void set_last_consumed_regst_acts(RegstAct* regst_act) {
    last_consumed_regst_acts_.push_back(regst_act);
  }

 private:
  int64_t topo_order_value_;
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

  void InitNodes();
  void InitEdges();
  void InitTopoOrderValue();
  void InitTaskId2TaskProto();
  void InitRegstActProduced7LastConsumedNode();
  void ForEachInEdge(const ChainActNode* node,
                     const std::function<void(const ChainActEdge*)>& Handler) const;
  void ForEachOutEdge(const ChainActNode* node,
                      const std::function<void(const ChainActEdge*)>& Handler) const;
  void TopoForEachChainActNode(std::list<ChainActNode*>& starts,
                               const std::function<void(ChainActNode*)>& Handler) const;
  void ForEachRegstActConsumerPathDuration(
      const std::function<void(int64_t, int64_t, double)>& Handler) const;
  void ForEachRegstActNodePathDuration(RegstActCtx* regst_act_ctx, const ChainActNode* node) const;
  void ForEachConsumerPathDuration(
      RegstActCtx* regst_act_ctx,
      const std::function<void(int64_t, int64_t, double)>& Handler) const;

  const Plan* plan_;
  std::unique_ptr<std::list<ActEvent>> act_events_;
  std::vector<const ChainActNode*> topo_nodes_;
  HashMap<int64_t, const TaskProto*> task_id2task_proto_;
  HashSet<const ActEvent*> act_event_with_consumer_;
  HashMap<const ActEvent*, ChainActNode*> act_event2chain_node_;
  HashMap<std::pair<int64_t, int64_t>, RegstAct*> regst_uid2regst_act_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_CHAIN_ACT_GRAPH_H_
