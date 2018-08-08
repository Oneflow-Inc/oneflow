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

struct ConsumerRegstInfos {
  ConsumerRegstInfos(const ActEvent* input_consumer,
                     const std::list<std::pair<int64_t, const ActEvent*>>& input_regst_id7producer)
      : consumer(input_consumer), regst_id7producer(input_regst_id7producer) {}

  const ActEvent* consumer;
  std::list<std::pair<int64_t, const ActEvent*>> regst_id7producer;
};

struct ConsumerRegstInfosCtx {
  ConsumerRegstInfosCtx(const ConsumerRegstInfos* input_consumer_regst_infos,
                        const ChainActNode* consumer_node)
      : consumer_regst_infos(input_consumer_regst_infos) {
    node2duration_to_consumer[consumer_node] = 0;
  }

  const ConsumerRegstInfos* consumer_regst_infos;
  HashMap<const ChainActNode*, double> node2duration_to_consumer;
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
  explicit ChainActNode(std::list<std::unique_ptr<ActEvent>>&& act_events);
  ~ChainActNode() = default;

  // ForEach
  void ForEachInEdge(const std::function<void(const ChainActEdge*)>& Handler) const;
  void ForEachOutEdge(const std::function<void(const ChainActEdge*)>& Handler) const;
  void ForEachActEvent(const std::function<void(const ActEvent*)>& Handler) const;
  void ForEachStartConsumerRegstInfos(
      const std::function<void(const ConsumerRegstInfos*)>& Handler) const;
  void ForEachStopConsumerRegstInfos(
      const std::function<void(const ConsumerRegstInfos*)>& Handler) const;

  // Getters
  int64_t act_id() const { return act_events_.front()->act_id(); }

  // Setters
  void AddStartConsumerRegstInfos(std::unique_ptr<ConsumerRegstInfos>&& consumed_regst_infos) {
    start_consumed_regst_infos_.push_back(std::move(consumed_regst_infos));
  }
  void AddStopConsumerRegstInfos(const ConsumerRegstInfos* consumed_regst_infos) {
    stop_consumed_regst_infos_.push_back(consumed_regst_infos);
  }

 private:
  std::list<std::unique_ptr<ActEvent>> act_events_;
  std::list<std::unique_ptr<ConsumerRegstInfos>> start_consumed_regst_infos_;
  std::list<const ConsumerRegstInfos*> stop_consumed_regst_infos_;
};

class ChainActGraph final : public Graph<const ChainActNode, const ChainActEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ChainActGraph);
  ChainActGraph(const Plan& plan, std::list<std::unique_ptr<ActEvent>>&& act_events);
  ~ChainActGraph() = default;

  // Getters
  const TaskProto& GetTaskProto(int64_t actor_id) const {
    return *task_id2task_proto_.at(actor_id);
  }

  // ForEach
  void ForEachRegstDescConsumerPathMeanDuration(
      const std::function<void(int64_t, int64_t, double)>& Handler) const;
  void ForEachRegstDescConsumerPathIIScale(
      const std::function<void(int64_t, int64_t, double)>& Handler) const;

  double CalcBaseII() const;

 private:
  bool IsActorWithConsumer(const ActEvent* act_event) const;
  const ChainActNode* Node4ActEvent(const ActEvent* act_event) const;
  void ForEachActEvent(const std::function<void(const ActEvent*)>& Handler) const;
  std::function<int64_t(const ChainActNode*)> MakeGetterTopoOrderValue4Node() const;

  void InitNodes(std::list<std::unique_ptr<ActEvent>>&& act_events,
                 HashMap<std::pair<int64_t, int64_t>, const ActEvent*>* regst_uid2producer);
  void InitEdges(const HashMap<std::pair<int64_t, int64_t>, const ActEvent*>& regst_uid2producer,
                 HashMap<const ActEvent*, std::list<std::pair<int64_t, const ActEvent*>>>*
                     consumer2regst_id7producer);
  void InitNodeStartConsumerRegstInfos(
      const HashMap<const ActEvent*, std::list<std::pair<int64_t, const ActEvent*>>>&
          consumer2regst_id7producer) const;
  void InitTopoNodes();
  void InitTaskId2TaskProto();
  void InitNodeStopConsumerRegstInfos() const;
  void DfsTopoForEachChainActNode(const std::function<void(const ChainActNode*)>& Handler) const;
  void ReverseDfsTopoForEachChainActNode(
      const std::function<void(const ChainActNode*)>& Handler) const;
  void ForEachRegstActConsumerPathDuration(
      const std::function<void(int64_t, int64_t, double)>& Handler) const;
  void CalcRegstActNodePathDuration(ConsumerRegstInfosCtx* ctx, const ChainActNode* node) const;
  void ForEachRegstDescDuration(const ConsumerRegstInfosCtx& ctx,
                                const std::function<void(int64_t, int64_t, double)>& Handler) const;

  const Plan* plan_;
  HashMap<int64_t, const TaskProto*> task_id2task_proto_;
  HashSet<const ActEvent*> actor_with_consumer_;
  HashMap<const ActEvent*, ChainActNode*> act_event2chain_node_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_CHAIN_ACT_GRAPH_H_
